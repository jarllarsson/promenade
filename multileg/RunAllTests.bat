@ECHO OFF
echo x64
cd bin/x64
Test_Debug
Test_Release
echo x86
cd ../x86
Test_Debug
Test_Release

pause