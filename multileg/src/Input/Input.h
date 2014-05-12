#pragma once


// =======================================================================================
//                                      Input
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # Input
/// 
/// 29-9-2013 Jarl Larsson - Updated 12-5-2014
///---------------------------------------------------------------------------------------


//////////////////////////////// OS Nuetral Headers ////////////////
#include <OISInputManager.h>
#include <OISException.h>
#include <OISKeyboard.h>
#include <OISMouse.h>
#include <OISJoyStick.h>
#include <OISEvents.h>

//Advanced Usage
#include <OISForceFeedback.h>

#include <iostream>
#include <vector>
#include <sstream>


////////////////////////////////////Needed Windows Headers////////////
#include <windows.h>
// #if defined OIS_WIN32_PLATFORM
// #  define WIN32_LEAN_AND_MEAN
// #  include "windows.h"
// #  ifdef min
// #    undef min
// #  endif
// #endif


#include "OISEventHandler.h"


using namespace OIS;


class Input
{
public:
	Input();
	virtual ~Input();
	
	OISEventHandler handler;

	void run();

	//-- Some local prototypes --//
	void doStartup(HWND p_hWnd);
	void handleNonBufferedKeys();
	void handleNonBufferedMouse();
	void handleNonBufferedJoy( JoyStick* js );
	bool hasJoysticks();

	// Return state of gamepad key
	// Returns false if gamepad doesn't exist, button doesn't exist or
	// if key isn't down
	bool gamepadButtonDown(int p_gamepadNum, int p_btnNum);
	
	static const char *g_DeviceType[6];

	InputManager*	g_InputManager;		//Our Input System
	Keyboard*		g_kb;				//Keyboard Device
	Mouse*			g_m;				//Mouse Device
	JoyStick*		g_joys[4];			//Support up to 4 controllers
protected:
private:
	bool m_hasJoysticks;
};