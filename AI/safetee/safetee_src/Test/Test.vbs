Option Explicit 
'--- start of vbslib include ------------------------------------------------------
'// �����́A�C�����Ȃ��ł��������B���L���C���֐�����X�N���v�g���L�q���Ă��������B
'// �l�r�I�t�B�X��R���p�C��������΁A���L�̂��Q���������� ���X�X�Ȃǂɂ���΁A�f�o�b�K���g���܂��B
'// �r���������߂��L�q����΁A�f�o�b�K�̓u���[�N���܂��B�ڂ����� vbslib �̐������̍Ō�́u�������Ƃ��́v�B
Dim  g_debug, g_debug_process, g_admin, g_vbslib_path, g_CommandPrompt, g_fs, g_sh, g_AppKey, g_Vers
If IsEmpty( g_fs ) Then
  Dim  g_MainPath, g_SrcPath : g_SrcPath = WScript.ScriptFullName : g_MainPath = g_SrcPath
  Set g_Vers = CreateObject("Scripting.Dictionary") : g_Vers.Add "vbslib", 3.0
  '--- start of parameters for vbslib include -------------------------------
  g_debug = 0          '// release:0, debug:99
  Sub SetupDebugTools() : set_input "" : SetBreakByFName Empty : End Sub
  g_vbslib_path = "vbslib\vbs_inc.vbs"
  g_CommandPrompt = 1
  '--- end of parameters for vbslib include ---------------------------------
  Dim  g_f, g_include_path, i : Set  g_fs = CreateObject( "Scripting.FileSystemObject" )
  Set  g_sh = WScript.CreateObject("WScript.Shell") : g_f = g_sh.CurrentDirectory
  g_sh.CurrentDirectory = g_fs.GetParentFolderName( WScript.ScriptFullName )
  For i = 20 To 1 Step -1 : If g_fs.FileExists(g_vbslib_path) Then  Exit For
  g_vbslib_path = "..\" + g_vbslib_path  : Next
  If g_fs.FileExists(g_vbslib_path) Then  g_vbslib_path = g_fs.GetAbsolutePathName( g_vbslib_path )
  g_sh.CurrentDirectory = g_f
  If i=0 Then WScript.Echo "Not found " + g_fs.GetFileName( g_vbslib_path ) +vbCR+vbLF+ "Let's download vbslib "&g_Vers.Item("vbslib")&" and Copy vbslib folder." : WScript.Quit 1
  Set g_f = g_fs.OpenTextFile( g_vbslib_path ): Execute g_f.ReadAll() : g_f = Empty
  If ResumePush Then  On Error Resume Next
    If IsDefined("main2") Then  Set g_f=CreateObject("Scripting.Dictionary") :_
      Set g_AppKey = new AppKeyClass : main2  g_f, g_AppKey.SetKey( new AppKeyClass )  Else _
      Set g_AppKey = new AppKeyClass : g_AppKey.SetKey( new AppKeyClass ) : main
  g_f = Empty : ResumePop : On Error GoTo 0
End If
'--- end of vbslib include --------------------------------------------------------


Sub main2( Opt, AppKey )
  set_input "" : SkipToSection Empty
  RunTestPrompt  AppKey.NewWritable( Array( _
    GetAbsPath( "..\..\safetee.exe", g_fs.GetParentFolderName( WScript.ScriptFullName ) ), _
    GetAbsPath( "..\Debug", g_fs.GetParentFolderName( WScript.ScriptFullName ) ), _
    GetAbsPath( "..\Release", g_fs.GetParentFolderName( WScript.ScriptFullName ) )  ) )
End Sub


Sub  test_current( tests )
End Sub


Sub  test_build( tests )
  devenv_rebuild  "..\safetee.sln", "Debug"
  devenv_rebuild  "..\safetee.sln", "Release"
  Pass
End Sub


Sub  test_setup( tests )
  Pass
End Sub


Sub  test_start( tests )
  ManualTest  "safetee Test1...bat"
  Pass
End Sub


Sub  test_check( tests )
  copy  "..\Release\safetee.exe", "..\..\safetee.exe"
  Pass
End Sub


Sub  test_clean( tests )
  devenv_clean   "..\safetee.sln"
  Pass
End Sub


 
