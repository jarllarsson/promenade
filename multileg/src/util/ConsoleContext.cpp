#include "ConsoleContext.h"

bool	ConsoleContext::consoleOpened = false;
bool	ConsoleContext::clearOnRefresh = false;
HANDLE	ConsoleContext::console = NULL;
bool    ConsoleContext::canWrite = false;
string  ConsoleContext::buffer = "";
float   ConsoleContext::refreshRate = 0.5f;
float   ConsoleContext::tick = 0.0f;


void ConsoleContext::init()
{
	if (!consoleOpened)
	{
		AllocConsole();
		console = GetStdHandle(STD_OUTPUT_HANDLE);
		WriteConsole(console,"\1 ~DEBUG~ \1\n\n",13,NULL,NULL);
		consoleOpened=true;
		SetConsoleTextAttribute(console,FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		buffer="";
		canWrite=false;
		tick=0.0f;
	}
}

void ConsoleContext::end()
{
	if (consoleOpened)
	{
		FreeConsole();
		consoleOpened=false;
		console = NULL;
	}
}


//
void ConsoleContext::addMsg(const string& msg,bool onRefresh)
{
	if (consoleOpened && (!onRefresh || (onRefresh && canWrite)) )
	{
		buffer+=msg;
	}
}


void ConsoleContext::refreshConsole(float dt)
{
	if (consoleOpened && console!=INVALID_HANDLE_VALUE && console!=NULL)
	{
		canWrite=false;
		tick+=dt;
		if (tick>=refreshRate)
		{
			// flush (does not work when running x64 build)
#ifndef X64
			if (clearOnRefresh)
			{
				COORD origin = {0,0};
				FillConsoleOutputCharacter(console, '\0', 
					MAXLONG, origin, NULL);
				// write header
				SetConsoleCursorPosition(console,origin);
				WriteConsole(console,"===========\nI  DEBUG  I\n===========\n",37,NULL,NULL);
			}
#endif
			// write debug info
			buffer = buffer.substr(0,800);
			WriteConsole(console,buffer.c_str(),buffer.size(),NULL,NULL);

			//
			buffer="";
			canWrite=true;
			tick=0.0f;
		}
	}

}

void ConsoleContext::setTitle(const string& msg)
{
	SetConsoleTitle(msg.c_str());
}
