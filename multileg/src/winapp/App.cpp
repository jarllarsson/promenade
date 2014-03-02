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

#include <iostream> 
#include <amp.h> 
#include <ppl.h>

//#include "OISHelper.h"

using namespace std;


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

	while (!m_context->closeRequested() && run)
	{
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
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
			m_controller->setFovFromAngle(52.0f,m_graphicsDevice->getAspectRatio());
			m_controller->update((float)dt);

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
			for(unsigned int i = 0; i < 11; i++) 
				std::cout << static_cast<char>(av[i]); 


			// ====================================================

			m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);		// Run passes
			m_graphicsDevice->flipBackBuffer();										// Flip!
			// ---------------------------------------------------------------------------------------------
		}
	}
}
