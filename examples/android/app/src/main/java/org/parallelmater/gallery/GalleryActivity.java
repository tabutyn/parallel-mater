// SPDX-License-Identifier: MIT
package org.parallelmater.gallery;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Insets;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;
import org.json.JSONArray;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Native gallery client. Physics, drawing and resource lifetime use one worker. */
public final class GalleryActivity extends Activity implements SurfaceHolder.Callback {
    static { System.loadLibrary("parallel_mater_gallery"); }
    private static native long nativeCreate(String assets, String manifest, String profiles,
        String model, String cache);
    private static native void nativeDestroy(long handle);
    private static native void nativeSurface(long handle, Surface surface, int width, int height);
    private static native String nativeFrame(long handle);
    private static native String nativeDetails(long handle);
    private static native void nativeCommand(long handle, int command, float x, float y);
    private static native int nativeTargetFps(long handle);
    private static native String[] nativeSceneNames();
    private static native double[] nativeQualifyScene(long handle, int scene, int frames);
    private static native String nativeRuntimeInfo(long handle);

    private static final String[] NAMES = nativeSceneNames();
    private static final String[] DESCRIPTIONS = {"Rigid-body collisions", "Fixed joints",
        "Point constraints", "Rotating and sliding joints", "Linear motion",
        "Translation and rotation limits", "Liquid particles", "Liquid–rigid interaction"};
    private static final int INK = Color.rgb(233, 243, 249);
    private static final int MUTED = Color.rgb(163, 184, 201);
    private static final int ACCENT = Color.rgb(102, 211, 246);
    private final HandlerThread thread = new HandlerThread("ParallelMaterGallery");
    private Handler worker;
    private TextView status, steering;
    private Button pause, release, scenes;
    // Native handles and frame scheduling belong to the worker thread.
    private long handle;
    private boolean active, surfaceReady, failed;
    private int frameInterval = 33;
    private long lastStatus;
    private int backgrounds, foregrounds;
    private String diagnosticNonce = "";
    // UI state is only accessed on the main thread.
    private boolean paused, released;
    private int selected;
    private float lastX, lastY;
    private ScaleGestureDetector scale;

    private final Runnable frame = new Runnable() {
        @Override public void run() {
            if (!active || !surfaceReady || failed || handle == 0) return;
            long start = SystemClock.uptimeMillis();
            try {
                String[] state = nativeFrame(handle).split("\\n", 2);
                String text = state[0];
                if (start - lastStatus >= 500) {
                    lastStatus = start;
                    runOnUiThread(() -> {
                        if (!status.getText().toString().equals(text)) status.setText(text);
                        if (state.length > 1) steering.setText(state[1]);
                    });
                }
            } catch (RuntimeException error) { showFailure(error); }
            if (!failed) worker.postDelayed(this,
                Math.max(1, frameInterval - (SystemClock.uptimeMillis() - start)));
        }
    };

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        thread.start();
        worker = new Handler(thread.getLooper());
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(11, 20, 29));
        SurfaceView surface = new SurfaceView(this);
        surface.getHolder().addCallback(this);
        root.addView(surface, new FrameLayout.LayoutParams(-1, -1));
        FrameLayout overlay = new FrameLayout(this);
        root.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        overlay.setOnApplyWindowInsetsListener((view, insets) -> {
            int left, top, right, bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                left = bars.left; top = bars.top; right = bars.right; bottom = bars.bottom;
            } else {
                left = insets.getSystemWindowInsetLeft(); top = insets.getSystemWindowInsetTop();
                right = insets.getSystemWindowInsetRight(); bottom = insets.getSystemWindowInsetBottom();
            }
            view.setPadding(left + dp(12), top + dp(10), right + dp(12), bottom + dp(10));
            return insets;
        });
        LinearLayout top = column();
        LinearLayout toolbar = row();
        TextView brand = label("Parallel Mater", 13, MUTED);
        brand.setTypeface(null, Typeface.BOLD);
        brand.setShadowLayer(dp(2), 0, dp(1), Color.BLACK);
        brand.setPadding(dp(10), 0, dp(16), 0);
        // Keep essential controls available on narrower landscape displays.
        if (getResources().getConfiguration().screenWidthDp >= 720)
            toolbar.addView(brand, new LinearLayout.LayoutParams(-2, dp(48)));
        scenes = button(NAMES[0] + "  ▾", this::chooseScene, false);
        scenes.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
        scenes.setTextSize(16);
        toolbar.addView(scenes, new LinearLayout.LayoutParams(dp(
            getResources().getConfiguration().screenWidthDp >= 720 ? 260 : 220), dp(48)));
        toolbar.addView(new View(this), new LinearLayout.LayoutParams(0, 1, 1));
        pause = button("Pause", () -> {
            paused = !paused;
            pause.setText(paused ? "Resume" : "Pause");
            command(2, paused ? 1 : 0, 0);
        }, true);
        addButton(toolbar, pause, 80);
        addButton(toolbar, button("Reset", this::resetScene, false), 72);
        addButton(toolbar, button("Info", this::showInfo, false), 60);
        top.addView(toolbar);
        status = label("Loading scene…", 12, INK);
        status.setPadding(dp(12), dp(5), dp(12), dp(5));
        status.setBackground(panel(false));
        LinearLayout.LayoutParams statusLayout = new LinearLayout.LayoutParams(-2, -2);
        statusLayout.topMargin = dp(8);
        top.addView(status, statusLayout);
        overlay.addView(top, new FrameLayout.LayoutParams(-1, -2, Gravity.TOP));

        LinearLayout movement = column();
        steering = label("Push objects", 11, INK);
        steering.setPadding(dp(10), dp(4), dp(10), dp(4));
        steering.setBackground(panel(false));
        LinearLayout.LayoutParams steeringLayout = new LinearLayout.LayoutParams(-2, -2);
        steeringLayout.bottomMargin = dp(6);
        steeringLayout.leftMargin = dp(6);
        movement.addView(steering, steeringLayout);
        LinearLayout arrows = row();
        addButton(arrows, direction("←", "Steer left", -1, 0), 48);
        addButton(arrows, direction("↑", "Steer forward", 0, 1), 48);
        addButton(arrows, direction("↓", "Steer back", 0, -1), 48);
        addButton(arrows, direction("→", "Steer right", 1, 0), 48);
        // The wrapping column must measure the row's buttons, not only its label.
        movement.addView(arrows, new LinearLayout.LayoutParams(-2, -2));
        overlay.addView(movement, new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.START));

        LinearLayout camera = row();
        release = button("Release", () -> {
            released = !released;
            release.setText(released ? "Restore" : "Release");
            command(6, 0, 0);
        }, false);
        release.setVisibility(View.GONE);
        addButton(camera, release, 84);
        addButton(camera, button("Focus", () -> command(8, 0, 0), false), 72);
        Button zoomOut = button("−", () -> command(5, -1, 0), false);
        zoomOut.setContentDescription("Zoom out");
        Button zoomIn = button("+", () -> command(5, 1, 0), false);
        zoomIn.setContentDescription("Zoom in");
        addButton(camera, zoomOut, 48);
        addButton(camera, zoomIn, 48);
        overlay.addView(camera, new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.END));
        setContentView(root);
        immersive();
        scale = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector detector) {
                command(5, (float)(Math.log(detector.getScaleFactor()) * 10), 0);
                return true;
            }
        });
        surface.setOnTouchListener((view, event) -> {
            scale.onTouchEvent(event);
            if (event.getActionMasked() == MotionEvent.ACTION_MOVE &&
                !scale.isInProgress() && event.getPointerCount() == 1) {
                float density = getResources().getDisplayMetrics().density;
                command(4, (event.getX() - lastX) / density, (event.getY() - lastY) / density);
            }
            lastX = event.getX(); lastY = event.getY();
            if (event.getActionMasked() == MotionEvent.ACTION_UP) view.performClick();
            return true;
        });
        // Enqueue initialization before callbacks or menu selections can use native state.
        worker.post(() -> {
            try {
                File files = getFilesDir();
                copyAssets("scenes", new File(files, "scenes"));
                copyAssets("config", new File(files, "config"));
                JSONObject driver = new JSONObject();
                driver.put("library_path", getApplicationInfo().nativeLibraryDir +
                    "/libparallel_mater_driver_slang_" + BuildConfig.PHYSICS_BACKEND + ".so");
                driver.put("api_version", "0.2.0");
                driver.put("library_arch", "64");
                JSONObject manifest = new JSONObject();
                manifest.put("file_format_version", "1.0.0");
                manifest.put("PMD", driver);
                File manifestFile = new File(files, "driver.json");
                try (FileOutputStream out = new FileOutputStream(manifestFile)) {
                    out.write(manifestBytes(manifest));
                }
                JSONObject layer = new JSONObject();
                layer.put("name", "PM_LAYER_KHRONOS_validation");
                layer.put("library_path", getApplicationInfo().nativeLibraryDir +
                    "/libparallel_mater_layer_validation.so");
                layer.put("api_version", "0.2.0");
                JSONObject layerManifest = new JSONObject();
                layerManifest.put("file_format_version", "1.0.0");
                layerManifest.put("layer", layer);
                try (FileOutputStream out = new FileOutputStream(new File(files, "layer.json"))) {
                    out.write(manifestBytes(layerManifest));
                }
                handle = nativeCreate(new File(files, "scenes").getPath(), manifestFile.getPath(),
                    new File(files, "config/device-profiles.json").getPath(), Build.MODEL,
                    new File(getCacheDir(), "vulkan").getPath());
                frameInterval = Math.max(1, 1000 / nativeTargetFps(handle));
                runOnUiThread(() -> {
                    status.setText("Ready");
                    applyDebugRequest(getIntent());
                });
                restartFrames();
            } catch (Exception error) { showFailure(error); }
        });
    }

    private void immersive() {
        if (Build.VERSION.SDK_INT >= 30) {
            getWindow().setDecorFitsSystemWindows(false);
            WindowInsetsController controller = getWindow().getInsetsController();
            if (controller != null) {
                controller.hide(WindowInsets.Type.systemBars());
                controller.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        } else {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        }
    }
    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (focused) immersive();
    }
    private void chooseScene() {
        command(3, 0, 0);
        ScrollView scroll = new ScrollView(this);
        LinearLayout grid = column();
        grid.setPadding(dp(16), dp(8), dp(16), dp(8));
        scroll.addView(grid);
        AlertDialog dialog = new AlertDialog.Builder(this).setTitle("Choose a scene")
            .setView(scroll).setNegativeButton("Close", null).create();
        for (int rowIndex = 0; rowIndex < 4; ++rowIndex) {
            LinearLayout row = row();
            for (int columnIndex = 0; columnIndex < 2; ++columnIndex) {
                int index = rowIndex * 2 + columnIndex;
                LinearLayout card = column();
                card.setPadding(dp(12), dp(8), dp(12), dp(8));
                card.setBackground(panel(index == selected));
                card.addView(label(NAMES[index], 15, index == selected ? Color.rgb(10, 29, 40) : INK));
                card.addView(label(DESCRIPTIONS[index], 11, index == selected ? Color.rgb(27, 63, 79) : MUTED));
                card.setContentDescription(NAMES[index] + ", " + DESCRIPTIONS[index]);
                card.setFocusable(true);
                card.setOnClickListener(view -> {
                    selectScene(index);
                    dialog.dismiss();
                });
                LinearLayout.LayoutParams layout = new LinearLayout.LayoutParams(0, dp(58), 1);
                layout.setMargins(dp(4), dp(4), dp(4), dp(4));
                row.addView(card, layout);
            }
            grid.addView(row);
        }
        dialog.show();
        styleDialog(dialog);
        dialog.getWindow().setLayout(Math.min(dp(620), getWindow().getDecorView().getWidth() - dp(48)),
            Math.min(dp(420), getWindow().getDecorView().getHeight() - dp(24)));
    }
    private void selectScene(int index) {
        if (index < 0 || index >= NAMES.length) return;
        selected = index;
        scenes.setText(NAMES[index] + "  ▾");
        scenes.setContentDescription("Choose scene: " + NAMES[index]);
        released = false;
        release.setText("Release");
        release.setVisibility(index == 2 ? View.VISIBLE : View.GONE);
        status.setText("Loading scene…");
        command(0, index, 0);
    }
    // Repeatable device qualification uses the same commands as the UI.
    // Release builds ignore these diagnostic extras.
    private void applyDebugRequest(Intent intent) {
        if (!BuildConfig.DEBUG || intent == null) return;
        if (intent.hasExtra("pm_qualify_frames")) {
            final int count = intent.getIntExtra("pm_qualify_frames", 120);
            final String nonce = intent.getStringExtra("pm_nonce");
            intent.removeExtra("pm_qualify_frames"); intent.removeExtra("pm_nonce");
            worker.post(() -> qualify(count, nonce == null ? "" : nonce));
        }
        if (intent.hasExtra("pm_scene")) selectScene(intent.getIntExtra("pm_scene", 0));
        String action = intent.getStringExtra("pm_action");
        if ("reset".equals(action)) resetScene();
        else if ("pause".equals(action) || "resume".equals(action)) {
            paused = "pause".equals(action);
            pause.setText(paused ? "Resume" : "Pause");
            command(2, paused ? 1 : 0, 0);
        } else if ("focus".equals(action)) command(8, 0, 0);
        else if ("info".equals(action)) showInfo();
        intent.removeExtra("pm_scene");
        intent.removeExtra("pm_action");
    }
    private void writeReport(String name, JSONObject report) throws Exception {
        android.util.AtomicFile file = new android.util.AtomicFile(new File(getFilesDir(), name));
        FileOutputStream stream = file.startWrite();
        try {
            stream.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
            file.finishWrite(stream);
        } catch (Exception error) { file.failWrite(stream); throw error; }
    }
    private void writeRuntimeInfo() {
        if (!BuildConfig.DEBUG || handle == 0) return;
        try {
            JSONObject report = new JSONObject(nativeRuntimeInfo(handle));
            report.put("nonce", diagnosticNonce).put("backgrounds", backgrounds).put("foregrounds", foregrounds);
            writeReport("runtime-info.json", report);
        } catch (Exception error) { showFailure(error); }
    }
    private void qualify(int count, String nonce) {
        if (handle == 0) return;
        try {
            if (count < 1 || count > 100000) throw new IllegalArgumentException("Qualification frames must be 1..100000");
            diagnosticNonce = nonce;
            worker.removeCallbacks(frame);
            JSONArray results = new JSONArray();
            JSONObject report = new JSONObject(nativeRuntimeInfo(handle));
            report.put("nonce", nonce).put("frames_per_scene", count).put("scenes", results);
            for (int i = 0; i < NAMES.length; ++i) {
                JSONObject result = new JSONObject().put("scene", NAMES[i]).put("steps", count);
                try {
                    double[] stats = nativeQualifyScene(handle, i, count);
                    result.put("passed", true).put("mean_ms", stats[0]).put("p95_ms", stats[1])
                        .put("timing_window_frames", stats[2]).put("session_controls", true);
                } catch (RuntimeException error) { result.put("passed", false).put("error", error.toString()); }
                results.put(result);
                report.put("completed", i + 1 == NAMES.length);
                writeReport("qualification.json", report);
            }
            nativeCommand(handle, 0, 0, 0);
            writeRuntimeInfo();
            runOnUiThread(() -> { selected = 0; scenes.setText(NAMES[0] + "  ▾"); });
            restartFrames();
        } catch (Exception error) { showFailure(error); }
    }
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        applyDebugRequest(intent);
    }
    private void resetScene() {
        released = false;
        release.setText("Release");
        command(1, 0, 0);
    }
    private void showInfo() {
        command(3, 0, 0);
        int sceneIndex = selected;
        worker.post(() -> {
            if (handle == 0) return;
            try {
                String details = nativeDetails(handle);
                runOnUiThread(() -> {
                    if (isFinishing() || isDestroyed()) return;
                    AlertDialog dialog = new AlertDialog.Builder(this).setTitle(NAMES[sceneIndex])
                        .setMessage(DESCRIPTIONS[sceneIndex] + "\n\n" + details +
                            "\n\nDrag to orbit · Pinch to zoom\nFocus reframes the current action.\nHold arrows to " +
                            steering.getText().toString().toLowerCase(java.util.Locale.ROOT) + ".")
                        .setPositiveButton("Done", null).show();
                    styleDialog(dialog);
                });
            } catch (RuntimeException error) { showFailure(error); }
        });
    }
    private void command(int kind, float x, float y) {
        worker.post(() -> {
            if (handle == 0) return;
            try {
                nativeCommand(handle, kind, x, y);
                if (kind == 0 || kind == 1) { failed = false; restartFrames(); }
                lastStatus = 0;
            } catch (RuntimeException error) { showFailure(error); }
        });
    }
    private void styleDialog(AlertDialog dialog) {
        GradientDrawable background = new GradientDrawable();
        background.setColor(Color.rgb(17, 30, 42));
        background.setCornerRadius(dp(18));
        background.setStroke(dp(1), Color.rgb(51, 73, 92));
        dialog.getWindow().setBackgroundDrawable(background);
        for (int which : new int[]{AlertDialog.BUTTON_POSITIVE, AlertDialog.BUTTON_NEGATIVE}) {
            Button button = dialog.getButton(which);
            if (button != null) { button.setAllCaps(false); button.setTextColor(ACCENT); }
        }
    }
    private static byte[] manifestBytes(JSONObject manifest) {
        // Android escapes slashes; the native manifest reader expects literal paths.
        return manifest.toString().replace("\\/", "/").getBytes(StandardCharsets.UTF_8);
    }
    private void restartFrames() {
        worker.removeCallbacks(frame);
        if (active && surfaceReady && !failed && handle != 0) worker.post(frame);
    }
    private void showFailure(Exception error) {
        failed = true;
        worker.removeCallbacks(frame);
        android.util.Log.e("ParallelMater", "Gallery error", error);
        String recovery = handle == 0 ? " — restart app" : " — try Reset or another scene";
        runOnUiThread(() -> status.setText("Error: " + error.getMessage() + recovery));
    }
    private void copyAssets(String path, File directory) throws Exception {
        if (!directory.isDirectory() && !directory.mkdirs()) throw new java.io.IOException("Cannot create " + directory);
        String[] children = getAssets().list(path);
        if (children == null) throw new java.io.IOException("Missing gallery assets");
        byte[] bytes = new byte[65536];
        for (String name : children) {
            try (InputStream in = getAssets().open(path + "/" + name);
                 FileOutputStream out = new FileOutputStream(new File(directory, name))) {
                int count;
                while ((count = in.read(bytes)) != -1) out.write(bytes, 0, count);
            }
        }
    }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private LinearLayout row() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        return row;
    }
    private LinearLayout column() {
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        return column;
    }
    private TextView label(String text, int size, int color) {
        TextView view = new TextView(this);
        view.setText(text); view.setTextSize(size); view.setTextColor(color);
        view.setGravity(Gravity.CENTER_VERTICAL);
        return view;
    }
    private RippleDrawable panel(boolean accent) {
        GradientDrawable background = new GradientDrawable();
        background.setColor(accent ? ACCENT : Color.argb(238, 23, 40, 55));
        background.setCornerRadius(dp(12));
        background.setStroke(dp(1), accent ? ACCENT : Color.rgb(51, 73, 92));
        return new RippleDrawable(ColorStateList.valueOf(Color.argb(60, 255, 255, 255)), background, null);
    }
    private Button button(String text, Runnable action, boolean accent) {
        Button button = new Button(this);
        button.setText(text); button.setTextSize(13); button.setAllCaps(false);
        button.setTextColor(accent ? Color.rgb(10, 29, 40) : INK);
        button.setTypeface(null, Typeface.BOLD);
        button.setPadding(dp(12), 0, dp(12), 0);
        button.setMinWidth(0); button.setMinimumWidth(0);
        button.setMinHeight(dp(48));
        button.setBackground(panel(accent));
        button.setOnClickListener(view -> action.run());
        return button;
    }
    private void addButton(LinearLayout row, Button button, int width) {
        LinearLayout.LayoutParams layout = new LinearLayout.LayoutParams(dp(width), dp(48));
        layout.leftMargin = dp(6);
        row.addView(button, layout);
    }
    private Button direction(String text, String description, float x, float y) {
        Button button = button(text, () -> {}, false);
        button.setTextSize(22);
        button.setContentDescription(description);
        button.setOnTouchListener((view, event) -> {
            int action = event.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN) { command(3, x, y); view.setPressed(true); }
            else if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                command(3, 0, 0); view.setPressed(false);
                if (action == MotionEvent.ACTION_UP) view.performClick();
            }
            return true;
        });
        return button;
    }
    @Override public void surfaceCreated(SurfaceHolder holder) {}
    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Surface surface = holder.getSurface();
        worker.post(() -> {
            if (handle == 0) return;
            try { nativeSurface(handle, surface, width, height); surfaceReady = true; restartFrames(); }
            catch (RuntimeException error) { showFailure(error); }
        });
    }
    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        worker.post(() -> {
            surfaceReady = false; worker.removeCallbacks(frame);
            if (handle != 0) nativeSurface(handle, null, 0, 0);
        });
    }
    @Override protected void onResume() {
        super.onResume();
        worker.post(() -> {
            active = true;
            if (handle != 0) nativeCommand(handle, 9, 1, 0);
            ++foregrounds; writeRuntimeInfo();
            restartFrames();
        });
    }
    @Override protected void onPause() {
        worker.post(() -> {
            active = false; worker.removeCallbacks(frame);
            if (handle != 0) nativeCommand(handle, 9, 0, 0);
            ++backgrounds; writeRuntimeInfo();
        });
        super.onPause();
    }
    @Override protected void onDestroy() {
        worker.post(() -> {
            active = false; worker.removeCallbacks(frame);
            nativeDestroy(handle); handle = 0; thread.quitSafely();
        });
        super.onDestroy();
    }
}
