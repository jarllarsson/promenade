#pragma once


// =======================================================================================
//                                      OISHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # OISHelper
/// 
/// 29-9-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

/*
//////////////////////////////// OS Neutral Headers ////////////////
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


class OISHelper
{
public:
	OISHelper();
	virtual ~OISHelper();
	
	OISEventHandler handler;

	void run();

	//-- Some local prototypes --//
	void doStartup(HWND p_hWnd);
	void handleNonBufferedKeys();
	void handleNonBufferedMouse();
	void handleNonBufferedJoy( JoyStick* js );
	
	static const char *g_DeviceType[6];

	InputManager*	g_InputManager;		//Our Input System
	Keyboard*		g_kb;				//Keyboard Device
	Mouse*			g_m;				//Mouse Device
	JoyStick*		g_joys[4];			//Support up to 4 controllers
protected:
private:
};

*/