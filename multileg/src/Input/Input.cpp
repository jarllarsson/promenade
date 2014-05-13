#include "Input.h"
#include <DebugPrint.h>


const char* Input::g_DeviceType[6] = {"OISUnknown", "OISKeyboard", "OISMouse", "OISJoyStick",
	"OISTablet", "OISOther"};

Input::Input()
{
	g_InputManager = NULL;
	g_kb  = NULL;			
	g_m   = NULL;			
	m_hasJoysticks = false;
	for (int i = 0; i<4; i++) g_joys[i] = NULL;
}

Input::~Input()
{
	//Destroying the manager will cleanup unfreed devices
	if( g_InputManager )
		InputManager::destroyInputSystem(g_InputManager);
}

void Input::doStartup( HWND p_hWnd )
{
	ParamList pl;
	std::ostringstream wnd;
	wnd << (size_t)p_hWnd;
	pl.insert(std::make_pair( std::string("WINDOW"), wnd.str() ));


#if defined OIS_WIN32_PLATFORM
	pl.insert(std::make_pair(std::string("w32_mouse"), std::string("DISCL_FOREGROUND")));
	pl.insert(std::make_pair(std::string("w32_mouse"), std::string("DISCL_NONEXCLUSIVE")));
	pl.insert(std::make_pair(std::string("w32_keyboard"), std::string("DISCL_FOREGROUND")));
	pl.insert(std::make_pair(std::string("w32_keyboard"), std::string("DISCL_NONEXCLUSIVE")));
#elif defined OIS_LINUX_PLATFORM
	pl.insert(std::make_pair(std::string("x11_mouse_grab"), std::string("false")));
	pl.insert(std::make_pair(std::string("x11_mouse_hide"), std::string("false")));
	pl.insert(std::make_pair(std::string("x11_keyboard_grab"), std::string("false")));
	pl.insert(std::make_pair(std::string("XAutoRepeatOn"), std::string("true")));
#endif

	//This never returns null.. it will raise an exception on errors
	g_InputManager = InputManager::createInputSystem(pl);

	//Lets enable all addons that were compiled in:
	g_InputManager->enableAddOnFactory(InputManager::AddOn_All);

	std::stringstream ss;

	//Print debugging information
	unsigned int v = g_InputManager->getVersionNumber();
	ss << "\nOIS Version: " << (v>>16 ) << "." << ((v>>8) & 0x000000FF) << "." << (v & 0x000000FF)
		<< "\nRelease Name: " << g_InputManager->getVersionName()
		<< "\nManager: " << g_InputManager->inputSystemName()
		<< "\nTotal Keyboards: " << g_InputManager->getNumberOfDevices(OISKeyboard)
		<< "\nTotal Mice: " << g_InputManager->getNumberOfDevices(OISMouse)
		<< "\nTotal JoySticks: " << g_InputManager->getNumberOfDevices(OISJoyStick);
	m_hasJoysticks = g_InputManager->getNumberOfDevices(OISJoyStick)>0;

	DEBUGPRINT((ss.str().c_str()));
	ss.clear();

	//List all devices
	DeviceList list = g_InputManager->listFreeDevices();
	for( DeviceList::iterator i = list.begin(); i != list.end(); ++i )
	{
		ss << "\n\tDevice: " << g_DeviceType[i->first] << " Vendor: " << i->second;
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
	}

	g_kb = (Keyboard*)g_InputManager->createInputObject( OISKeyboard, true );
	g_kb->setEventCallback( &handler );

	g_m = (Mouse*)g_InputManager->createInputObject( OISMouse, true );
	g_m->setEventCallback( &handler );
	const MouseState &ms = g_m->getMouseState();
	ms.width = 100;
	ms.height = 100;

	try
	{
		//This demo uses at most 4 joysticks - use old way to create (i.e. disregard vendor)
		int numSticks = min(g_InputManager->getNumberOfDevices(OISJoyStick), 4);
		for( int i = 0; i < numSticks; ++i )
		{
			g_joys[i] = (JoyStick*)g_InputManager->createInputObject( OISJoyStick, true );
			g_joys[i]->setEventCallback( &handler );
			ss << "\n\nCreating Joystick " << (i + 1)
				<< "\n\tAxes: " << g_joys[i]->getNumberOfComponents(OIS_Axis)
				<< "\n\tSliders: " << g_joys[i]->getNumberOfComponents(OIS_Slider)
				<< "\n\tPOV/HATs: " << g_joys[i]->getNumberOfComponents(OIS_POV)
				<< "\n\tButtons: " << g_joys[i]->getNumberOfComponents(OIS_Button)
				<< "\n\tVector3: " << g_joys[i]->getNumberOfComponents(OIS_Vector3);
			DEBUGPRINT((ss.str().c_str()));
			ss.clear();
		}
	}
	catch(OIS::Exception &ex)
	{
		ss << "\nException raised on joystick creation: " << ex.eText << std::endl;
		DEBUGPRINT((ss.str().c_str()));
		ss.clear();
	}
}

void Input::handleNonBufferedKeys()
{
	std::stringstream ss;
	if( g_kb->isModifierDown(Keyboard::Shift) )
		ss << "\nShift is down..\n";
	if( g_kb->isModifierDown(Keyboard::Alt) )
		ss << "\nAlt is down..\n";
	if( g_kb->isModifierDown(Keyboard::Ctrl) )
		ss << "\nCtrl is down..\n";
	DEBUGPRINT((ss.str().c_str()));
	ss.clear();
}

void Input::handleNonBufferedMouse()
{
// 	std::stringstream ss;
// 	//Just dump the current mouse state
// 	const MouseState &ms = g_m->getMouseState();
// 	ss << "\nMouse: Abs(" << ms.X.abs << " " << ms.Y.abs << " " << ms.Z.abs
// 		<< ") B: " << ms.buttons << " Rel(" << ms.X.rel << " " << ms.Y.rel << " " << ms.Z.rel << ")";
// 	DEBUGPRINT((ss.str().c_str()));
// 	ss.clear();
}

void Input::handleNonBufferedJoy( JoyStick* js )
{
// 	std::stringstream ss;
// 	//Just dump the current joy state
// 	const JoyStickState &joy = js->getJoyStickState();
// 	for( unsigned int i = 0; i < joy.mAxes.size(); ++i )
// 	{
// 		ss << "\nAxis " << i << " X: " << joy.mAxes[i].abs;
// 		DEBUGPRINT((ss.str().c_str()));
// 		ss.clear();
// 	}
}

bool Input::gamepadButtonDown(int p_gamepadNum, int p_btnNum)
{
	bool result = false;
	JoyStick* joy = nullptr;
	if (hasJoysticks())
	{
		joy = g_joys[p_gamepadNum];
		if (joy != nullptr &&
			joy->getJoyStickState().mButtons.size() < (unsigned int)(p_btnNum + 1) &&
			joy->getJoyStickState().mButtons[p_btnNum])
		{
			result = true;
		}
	}
	return result;
}

void Input::run()
{
	if( g_kb )
	{
		g_kb->capture();
		if( !g_kb->buffered() )
			handleNonBufferedKeys();
	}

	if( g_m )
	{
		g_m->capture();
		if( !g_m->buffered() )
			handleNonBufferedMouse();
	}
	
	for( int i = 0; i < 4 ; ++i )
	{
		if( g_joys[i] )
		{
			g_joys[i]->capture();
			if( !g_joys[i]->buffered() )
				handleNonBufferedJoy( g_joys[i] );
		}
	}
}

bool Input::hasJoysticks()
{
	return m_hasJoysticks;
}