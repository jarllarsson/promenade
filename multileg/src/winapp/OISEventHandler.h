#pragma once

// =======================================================================================
//                                      OISEventHandler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # OISEventHandler
/// 
/// 29-9-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

/*
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
#include <DebugPrint.h>

using namespace OIS;

//////////// Common Event handler class ////////
class OISEventHandler : public KeyListener, public MouseListener, public JoyStickListener
{
public:
	OISEventHandler() {}
	~OISEventHandler() {}
	bool keyPressed( const KeyEvent &arg ) 
	{
		std::stringstream ss;
		ss << "\n KeyPressed {" << arg.key
			<< ", " << ((Keyboard*)(arg.device))->getAsString(arg.key)
			<< "} || Character (" << (char)arg.text << ")" << std::endl;
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool keyReleased( const KeyEvent &arg ) 
	{
		std::stringstream ss;
		if( arg.key == KC_ESCAPE || arg.key == KC_Q )
		ss << "\nKeyReleased {" << ((Keyboard*)(arg.device))->getAsString(arg.key) << "}\n";
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool mouseMoved( const MouseEvent &arg ) 
	{
		std::stringstream ss;
		const OIS::MouseState& s = arg.state;
		ss << "\nMouseMoved: Abs("
			<< s.X.abs << ", " << s.Y.abs << ", " << s.Z.abs << ") Rel("
			<< s.X.rel << ", " << s.Y.rel << ", " << s.Z.rel << ")";
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool mousePressed( const MouseEvent &arg, MouseButtonID id ) 
	{
		std::stringstream ss;
		const OIS::MouseState& s = arg.state;
		ss << "\nMouse button #" << id << " pressed. Abs("
			<< s.X.abs << ", " << s.Y.abs << ", " << s.Z.abs << ") Rel("
			<< s.X.rel << ", " << s.Y.rel << ", " << s.Z.rel << ")";
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool mouseReleased( const MouseEvent &arg, MouseButtonID id ) 
	{
		std::stringstream ss;
		const OIS::MouseState& s = arg.state;
		ss << "\nMouse button #" << id << " released. Abs("
			<< s.X.abs << ", " << s.Y.abs << ", " << s.Z.abs << ") Rel("
			<< s.X.rel << ", " << s.Y.rel << ", " << s.Z.rel << ")";
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool buttonPressed( const JoyStickEvent &arg, int button ) 
	{
		std::stringstream ss;
		ss << std::endl << arg.device->vendor() << ". Button Pressed # " << button;
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool buttonReleased( const JoyStickEvent &arg, int button ) 
	{
		std::stringstream ss;
		ss << std::endl << arg.device->vendor() << ". Button Released # " << button;
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
	bool axisMoved( const JoyStickEvent &arg, int axis )
	{
		std::stringstream ss;
		//Provide a little dead zone
		if( arg.state.mAxes[axis].abs > 2500 || arg.state.mAxes[axis].abs < -2500 )
		{	
			ss << std::endl << arg.device->vendor() << ". Axis # " << axis << " Value: " << arg.state.mAxes[axis].abs;
			DEBUGPRINT((ss.str().c_str()));
			ss.clear();
		}

		return true;
	}
	bool povMoved( const JoyStickEvent &arg, int pov )
	{
		std::stringstream ss;
		ss << std::endl << arg.device->vendor() << ". POV" << pov << " ";

		if( arg.state.mPOV[pov].direction & Pov::North ) //Going up
			ss << "North";
		else if( arg.state.mPOV[pov].direction & Pov::South ) //Going down
			ss << "South";

		if( arg.state.mPOV[pov].direction & Pov::East ) //Going right
			ss << "East";
		else if( arg.state.mPOV[pov].direction & Pov::West ) //Going left
			ss << "West";

		if( arg.state.mPOV[pov].direction == Pov::Centered ) //stopped/centered out
			ss << "Centered";
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}

	bool vector3Moved( const JoyStickEvent &arg, int index)
	{
		std::stringstream ss;
		ss.precision(2);
		ss.flags(std::ios::fixed | std::ios::right);
		ss << std::endl << arg.device->vendor() << ". Orientation # " << index 
			<< " X Value: " << arg.state.mVectors[index].x
			<< " Y Value: " << arg.state.mVectors[index].y
			<< " Z Value: " << arg.state.mVectors[index].z;
		ss.precision();
		ss.flags();
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
		return true;
	}
};
*/