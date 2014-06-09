@ECHO OFF
echo x64
cd bin/x64
Test_Debug -s
Test_Release -s
echo x86
cd ../x86
Test_Debug -s
Test_Release -s

pause