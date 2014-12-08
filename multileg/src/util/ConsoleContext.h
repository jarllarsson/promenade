#pragma once

#include <string>
#include <windows.h>
// =======================================================================================
//                                      ConsoleContext
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ConsoleContext
/// Detailed description.....
/// Created on: 8-12-2014 
///---------------------------------------------------------------------------------------


using namespace std;

class ConsoleContext
{
public:
	static void init();
	static void end();
	static void setTitle(const string& msg);
	static void addMsg(const string& msg,bool onRefresh);
	static void refreshConsole(float dt);
private:
	static bool		consoleOpened;
	static bool		clearOnRefresh;
	static HANDLE	console;
	static bool		canWrite;
	static string	buffer;
	static float	refreshRate;
	static float	tick;
};
