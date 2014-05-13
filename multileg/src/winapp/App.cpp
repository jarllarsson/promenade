#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>
#include <BufferFactory.h>

#include <Input.h>
#include <Util.h>

#include <ValueClamp.h>
#include "TempController.h"

#include <iostream> 
#include <amp.h> 
#include <ppl.h>



using namespace std;


const double App::DTCAP=0.5;

App::App( HINSTANCE p_hInstance )
{
	int width=1280,
		height=1024;
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


	fpsUpdateTick=0.0f;
	m_controller = new TempController();
	m_input = new Input();
	m_input->doStartup(m_context->getWindowHandle());
	//
	for (int x = 0; x < 100; x++)
	for (int y = 0; y < 10; y++)
	for (int z = 0; z < 100; z++)
	{
		glm::mat4 transMat = glm::translate(glm::mat4(1.0f),
			glm::vec3((float)x*2.0f -100.0f, (float)y*2.0f-100.0f, (float)z*2.0f));
		transMat = glm::transpose(transMat);
		m_instance.push_back(transMat);
	}
	m_instances = m_graphicsDevice->getBufferFactoryRef()->createMat4InstanceBuffer((void*)&m_instance[0], m_instance.size());
	m_vp = m_graphicsDevice->getBufferFactoryRef()->createMat4CBuffer();
}

App::~App()
{	
	//SAFE_DELETE(m_kernelDevice);
	SAFE_DELETE(m_graphicsDevice);
	SAFE_DELETE(m_context);
	SAFE_DELETE(m_input);
	SAFE_DELETE(m_controller);
	//
	delete m_instances;
	delete m_vp;
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

	while (!m_context->closeRequested() && run)
	{
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
			m_input->run();
			// apply resizing on graphics device if it has been triggered by the context
			if (m_context->isSizeDirty())
			{
				pair<int,int> sz=m_context->getSize();
				m_graphicsDevice->updateResolution(sz.first,sz.second);
			}

			// Get Delta time
			QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

			// Calculate delta time and fps
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

			// Clear render targets
			m_graphicsDevice->clearRenderTargets();									// Clear render targets

			// temp controller update code
			updateController((float)dt);
			m_controller->setFovFromAngle(52.0f,m_graphicsDevice->getAspectRatio());
			m_controller->update((float)dt);
			// Get camera info to buffer
			std::memcpy(&m_vp->accessBuffer, &m_controller->getViewProjMatrix(), sizeof(float)* 4 * 4);
			m_vp->update();

			// Run the devices
			// ---------------------------------------------------------------------------------------------
			
			int v[11] = {'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};

			// Serial (CPU)


			// PPL (CPU)
			int pplRes[11];
			concurrency::parallel_for(0, 11, [&](int n) {
				pplRes[n]=v[n]+1;
			});
			for(unsigned int i = 0; i < 11; i++) 
				std::cout << static_cast<char>(pplRes[i]); 

			// C++AMP (GPU)
			concurrency::array_view<int> av(11, v); 
			concurrency::parallel_for_each(av.extent, [=](concurrency::index<1> idx) restrict(amp) 
			{ 
				av[idx] += 1; 
			});


			// Print C++AMP
			for (unsigned int i = 0; i < 11; i++)
			{
				char ch = static_cast<char>(av[i]);
				DEBUGPRINT(( toString(ch).c_str() ));
			}
			DEBUGPRINT((string("\n").c_str()));


			// ====================================================

			m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);		// Run passes
			m_graphicsDevice->executeRenderPass(GraphicsDevice::P_WIREFRAMEPASS, m_vp, m_instances);
			m_graphicsDevice->flipBackBuffer();										// Flip!
			// ---------------------------------------------------------------------------------------------
		}
	}
}


__int64 getTimeStamp()
{
	__int64 stamp = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&stamp);
	return stamp;
}

__int64 getFrequency()
{
	__int64 freq = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	return freq;
}

void App::updateController(float p_dt)
{
	//m_controller->moveThrust(glm::vec3(0.0f, 0.0f, -0.8f));
	//m_controller->moveAngularThrust(glm::vec3(0.0f, 0.0f, 1.0f)*0.07f);
	// get joystate
	//Just dump the current joy state
	JoyStick* joy = nullptr;
	if (m_input->hasJoysticks()) 
		joy = m_input->g_joys[0];
	float thrustPow = 10.0f;
	// Thrust
	if (m_input->g_kb->isKeyDown(KC_LEFT) || m_input->g_kb->isKeyDown(KC_A))
		m_controller->moveThrust(glm::vec3(-1.0f,0.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_RIGHT) || m_input->g_kb->isKeyDown(KC_D))
		m_controller->moveThrust(glm::vec3(1.0f,0.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_UP) || m_input->g_kb->isKeyDown(KC_W))
		m_controller->moveThrust(glm::vec3(0.0f,1.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_DOWN) || m_input->g_kb->isKeyDown(KC_S))
		m_controller->moveThrust(glm::vec3(0.0f,-1.0f,0.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_SPACE))
		m_controller->moveThrust(glm::vec3(0.0f,0.0f,1.0f)*thrustPow);
	if (m_input->g_kb->isKeyDown(KC_B))
		m_controller->moveThrust(glm::vec3(0.0f,0.0f,-1.0f)*thrustPow);
	// Joy thrust
	if (joy!=nullptr)
	{
		const JoyStickState& js = joy->getJoyStickState();
		m_controller->moveThrust(glm::vec3((float)(invclampcap(js.mAxes[1].abs,-5000,5000))* 0.0001f,
			(float)(invclampcap(js.mAxes[0].abs,-5000,5000))*-0.0001f,
			(float)(js.mAxes[4].abs)*-0.0001f)*thrustPow);
	}
	
	
	// Angular thrust
	if (m_input->g_kb->isKeyDown(KC_Q) || (joy!=nullptr && joy->getJoyStickState().mButtons[4]))
		m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,-1.0f));
	if (m_input->g_kb->isKeyDown(KC_E) || (joy!=nullptr && joy->getJoyStickState().mButtons[5]))
		m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,1.0f));
	if (m_input->g_kb->isKeyDown(KC_T))
		m_controller->moveAngularThrust(glm::vec3(0.0f,1.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_R))
		m_controller->moveAngularThrust(glm::vec3(0.0f,-1.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_U))
		m_controller->moveAngularThrust(glm::vec3(1.0f,0.0f,0.0f));
	if (m_input->g_kb->isKeyDown(KC_J))
		m_controller->moveAngularThrust(glm::vec3(-1.0f,0.0f,0.0f));
	// Joy angular thrust
	if (joy!=nullptr)
	{
		const JoyStickState& js = joy->getJoyStickState();
		m_controller->moveAngularThrust(glm::vec3((float)(invclampcap(js.mAxes[2].abs,-5000,5000))*-0.00001f,
			(float)(invclampcap(js.mAxes[3].abs,-5000,5000))*-0.00001f,
			0.0f));
	}
	
	float mousemovemultiplier=0.002f;
	float mouseX=(float)m_input->g_m->getMouseState().X.rel*mousemovemultiplier;
	float mouseY=(float)m_input->g_m->getMouseState().Y.rel*mousemovemultiplier;
	if ((abs(mouseX)>0.0f || abs(mouseY)>0.0f) && m_input->g_m->getMouseState().buttonDown(MB_Middle))
	{
		m_controller->rotate(glm::vec3(clamp(mouseY,-1.0f,1.0f),clamp(mouseX,-1.0f,1.0f),0.0f));
	}
}