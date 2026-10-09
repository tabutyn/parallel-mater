// SPDX-License-Identifier: MIT
// Native Start menu entry. Keep the gallery's console interface for automation,
// but do not open a console window when launching the installed desktop app.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <filesystem>
#include <string>
#include <vector>

int WINAPI wWinMain(HINSTANCE,HINSTANCE,PWSTR arguments,int) {
    std::vector<wchar_t> module(32768);
    const DWORD length=GetModuleFileNameW(nullptr,module.data(),static_cast<DWORD>(module.size()));
    if(length==0 || length>=module.size()) return 1;
    const auto directory=std::filesystem::path(module.data()).parent_path();
    const auto executable=directory/L"parallel-mater-d3d12-gallery.exe";
    PWSTR local_app_data=nullptr;
    if(FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData,0,nullptr,&local_app_data))) return 1;
    const auto data_directory=std::filesystem::path(local_app_data)/L"Parallel-Mater";
    CoTaskMemFree(local_app_data);
    std::error_code error;
    std::filesystem::create_directories(data_directory,error);
    if(error) return 1;
    const auto log_path=data_directory/L"gallery.log";
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
    HANDLE log=CreateFileW(log_path.c_str(),FILE_APPEND_DATA,FILE_SHARE_READ|FILE_SHARE_WRITE,
                           &security,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,
                             &security,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(log==INVALID_HANDLE_VALUE || input==INVALID_HANDLE_VALUE) {
        if(log!=INVALID_HANDLE_VALUE) CloseHandle(log);
        if(input!=INVALID_HANDLE_VALUE) CloseHandle(input);
        MessageBoxW(nullptr,L"Could not open the Parallel-Mater log.",L"Parallel-Mater",MB_OK|MB_ICONERROR);
        return 1;
    }
    std::wstring command=L"\""+executable.wstring()+L"\"";
    if(arguments && *arguments) command+=L" "+std::wstring(arguments);
    STARTUPINFOW startup{};
    startup.cb=sizeof(startup);
    startup.dwFlags=STARTF_USESTDHANDLES;
    startup.hStdOutput=log; startup.hStdError=log; startup.hStdInput=input;
    PROCESS_INFORMATION process{};
    const BOOL created=CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,
        CREATE_NO_WINDOW,nullptr,data_directory.c_str(),&startup,&process);
    CloseHandle(log); CloseHandle(input);
    if(!created) {
        MessageBoxW(nullptr,L"Could not start the gallery. Reinstall Parallel-Mater.",L"Parallel-Mater",MB_OK|MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess,INFINITE);
    DWORD result=1;
    GetExitCodeProcess(process.hProcess,&result);
    CloseHandle(process.hProcess);
    if(result!=0) {
        const std::wstring message=L"The gallery exited with an error. Details:\n"+log_path.wstring();
        MessageBoxW(nullptr,message.c_str(),L"Parallel-Mater",MB_OK|MB_ICONERROR);
    }
    return static_cast<int>(result);
}
