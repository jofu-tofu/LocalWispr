; LocalWispr Inno Setup Script
; Creates a Windows installer with optional desktop shortcut and autostart

#define MyAppName "LocalWispr"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "LocalWispr"
#define MyAppURL "https://github.com/jofu-tofu/LocalWispr"
#define MyAppExeName "LocalWispr.exe"

[Setup]
; Unique application ID - DO NOT CHANGE after first release
AppId={{4A8F2B3C-1D5E-6F7A-8B9C-0D1E2F3A4B5C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Install location
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=..\dist
OutputBaseFilename=LocalWispr-Setup-{#MyAppVersion}

; Compression settings (lzma2 is best for size)
Compression=lzma2/ultra64
SolidCompression=yes

; Windows version requirements (Windows 10 1809 or later)
MinVersion=10.0.17763

; Installer appearance
WizardStyle=modern

; Privileges (install for current user only)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Uninstall settings
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Start {#MyAppName} when Windows starts"; GroupDescription: "Startup Options:"

[Files]
; Main executable and config
Source: "..\dist\LocalWispr\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\LocalWispr\config-defaults.toml"; DestDir: "{app}"; Flags: ignoreversion

; Internal dependencies
Source: "..\dist\LocalWispr\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

; Desktop shortcut (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Autostart entry (optional)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
; Launch after install
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Check if the application is running before uninstall
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Result := '';
  // Try to stop the app if running (silently)
  Exec('taskkill', '/f /im LocalWispr.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
