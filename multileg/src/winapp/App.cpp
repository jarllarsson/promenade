#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>

#include <Util.h>

//#include "KernelDevice.h"
//#include "KernelException.h"

#include <ValueClamp.h>
#include "TempController.h"


//#include "OISHelper.h"


const double App::DTCAP=0.5;

App::App( HINSTANCE p_hInstance )
{
	int width=600,
		height=400;
	bool windowMode=true;
	// Context
	try
	{
		m_context = new Context(p_hInstance,"multileg",width,height);
	}
	catch (ContextException& e)
	{
		DEBUGWARNING((e.what()));
	}	
	
	// Graphics
	try
	{
		m_graphicsDevice = new GraphicsDevice(m_context->getWindowHandle(),width,height,windowMode);
	}
	catch (GraphicsException& e)
	{
		DEBUGWARNING((e.what()));
	}

	// Kernels
	/*
	try
	{
		m_kernelDevice = new KernelDevice(m_graphicsDevice->getDevicePointer());
		m_kernelDevice->registerCanvas(m_graphicsDevice->getInteropCanvasHandle());
	}
	catch (KernelException& e)
	{
		DEBUGWARNING((e.what()));
	}
	*/

	//

	fpsUpdateTick=0.0f;
	m_controller = new TempController();
	//m_input = new OISHelper();
	//m_input->doStartup(m_context->getWindowHandle());
}

App::~App()
{	
	//SAFE_DELETE(m_kernelDevice);
	SAFE_DELETE(m_graphicsDevice);
	SAFE_DELETE(m_context);
	//SAFE_DELETE(m_input);
	SAFE_DELETE(m_controller);
}

void App::run()
{
	// Set up windows timer
	__int64 countsPerSec = 0;
	__int64 currTimeStamp = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	double secsPerCount = 1.0f / (float)countsPerSec;

	double dt = 0.0;
	double fps = 0.0f;
	__int64 m_prevTimeStamp = 0;

	QueryPerformanceCounter((LARGE_INTEGER*)&m_prevTimeStamp);
	QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

	MSG msg = {0};

	// secondary run variable
	// lets non-context systems quit the program
	bool run=true;

	int shadowMode=0;
	int debugDrawMode=0;

	while (!m_context->closeRequested() && run)
	{
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
			/*
			m_input->run();
			// Thrust
			if (m_input->g_kb->isKeyDown(KC_LEFT) || m_input->g_kb->isKeyDown(KC_A))
				m_controller->moveThrust(glm::vec3(-1.0f,0.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_RIGHT) || m_input->g_kb->isKeyDown(KC_D))
				m_controller->moveThrust(glm::vec3(1.0f,0.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_UP) || m_input->g_kb->isKeyDown(KC_W))
				m_controller->moveThrust(glm::vec3(0.0f,1.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_DOWN) || m_input->g_kb->isKeyDown(KC_S))
				m_controller->moveThrust(glm::vec3(0.0f,-1.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_SPACE))
				m_controller->moveThrust(glm::vec3(0.0f,0.0f,1.0f));
			if (m_input->g_kb->isKeyDown(KC_B))
				m_controller->moveThrust(glm::vec3(0.0f,0.0f,-1.0f));
			// Angular thrust
			if (m_input->g_kb->isKeyDown(KC_Q))
				m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,-1.0f));
			if (m_input->g_kb->isKeyDown(KC_E))
				m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,1.0f));
			if (m_input->g_kb->isKeyDown(KC_T))
				m_controller->moveAngularThrust(glm::vec3(0.0f,1.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_R))
				m_controller->moveAngularThrust(glm::vec3(0.0f,-1.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_U))
				m_controller->moveAngularThrust(glm::vec3(1.0f,0.0f,0.0f));
			if (m_input->g_kb->isKeyDown(KC_J))
				m_controller->moveAngularThrust(glm::vec3(-1.0f,0.0f,0.0f));
			// Settings
			if (m_input->g_kb->isKeyDown(KC_B)) // Debug blocks
				debugDrawMode=1;
			if (m_input->g_kb->isKeyDown(KC_N)) // Debug off
				debugDrawMode=0;
			if (m_input->g_kb->isKeyDown(KC_0)) // Shadow off
				shadowMode=0;
			if (m_input->g_kb->isKeyDown(KC_1)) // Shadow on (hard shadows)
				shadowMode=1;
			if (m_input->g_kb->isKeyDown(KC_2)) // Shadow on (soft shadows fidelity=2)
				shadowMode=2;
			if (m_input->g_kb->isKeyDown(KC_3)) // Shadow on (soft shadows fidelity=5)
				shadowMode=5;
			if (m_input->g_kb->isKeyDown(KC_4)) // Shadow on (soft shadows fidelity=10)
				shadowMode=10;
			if (m_input->g_kb->isKeyDown(KC_5)) // Shadow on (soft shadows fidelity=15)
				shadowMode=15;
			if (m_input->g_kb->isKeyDown(KC_6)) // Shadow on (soft shadows fidelity=20)
				shadowMode=20;

			float mousemovemultiplier=0.001f;
 			float mouseX=(float)m_input->g_m->getMouseState().X.rel*mousemovemultiplier;
 			float mouseY=(float)m_input->g_m->getMouseState().Y.rel*mousemovemultiplier;
 			if (abs(mouseX)>0.0f || abs(mouseY)>0.0f)
 			{
 				m_controller->rotate(glm::vec3(clamp(-mouseY,-1.0f,1.0f),clamp(-mouseX,-1.0f,1.0f),0.0f));
 			}
			*/
			// apply resizing on graphics device if it has been triggered by the context
			if (m_context->isSizeDirty())
			{
				pair<int,int> sz=m_context->getSize();
				m_graphicsDevice->updateResolution(sz.first,sz.second);
			}

			// Get Delta time
			QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

			dt = (currTimeStamp - m_prevTimeStamp) * secsPerCount;
			fps = 1.0f/dt;
			
			dt = clamp(dt,0.0,DTCAP);
			m_prevTimeStamp = currTimeStamp;

			fpsUpdateTick-=(float)dt;
			if (fpsUpdateTick<=0.0f)
			{
				m_context->updateTitle((" | FPS: "+toString((int)fps)).c_str());
				//DEBUGPRINT((("\n"+toString(dt)).c_str())); 
				fpsUpdateTick=0.3f;
			}

			m_graphicsDevice->clearRenderTargets();									// Clear render targets

			// temp controller update code
			//m_controller->setFovFromAngle(52.0f,m_graphicsDevice->getAspectRatio());
			//m_controller->update((float)dt);

			// Run the devices
			// ---------------------------------------------------------------------------------------------
			//m_kernelDevice->update((float)dt,m_controller,debugDrawMode,shadowMode);	// Update kernel data



			//m_kernelDevice->executeKernelJob((float)dt,KernelDevice::J_RAYTRACEWORLD);		// Run kernels

			m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);		// Run passes
			m_graphicsDevice->flipBackBuffer();										// Flip!
			// ---------------------------------------------------------------------------------------------
		}
	}
}
