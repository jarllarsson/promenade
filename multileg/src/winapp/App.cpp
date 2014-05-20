#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>
#include <BufferFactory.h>

#include <Input.h>
#include <Util.h>
#include <MeasurementBin.h>

#include <ValueClamp.h>
#include "TempController.h"

#include <iostream> 
#include <amp.h> 
#include <ppl.h>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>


#include "PositionSystem.h"
#include "RigidBodySystem.h"
#include "RenderSystem.h"
#include "ControllerSystem.h"
#include "PhysicsWorldHandler.h"



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


	m_fpsUpdateTick=0.0f;
	m_controller = new TempController(0.0f,10.0f,-50.0f,0.0f);
	m_input = new Input();
	m_input->doStartup(m_context->getWindowHandle());
	m_timeScale = 1.0f;
	m_timeScaleToggle = false;
	m_timePauseStepToggle = false;
	//
	//for (int x = 0; x < 10; x++)
	//for (int z = 0; z < 10; z++)
	//{
	//	glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 1.0f, 20.0f));
	//	glm::mat4 transMat = glm::translate(scaleMat, 
	//		glm::vec3((float)x*5.0f -50.0f, 0.0f, (float)z*5.0f - 50.0f));
	//	transMat = glm::transpose(transMat);
	//	m_instanceOrigins.push_back(transMat);
	//}
	//m_instances = m_graphicsDevice->getBufferFactoryRef()->createMat4InstanceBuffer((void*)&m_instanceOrigins[0], (unsigned int)m_instanceOrigins.size());
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
	//delete m_instances;
	delete m_vp;
}

void App::run()
{
	// Set up windows timer
	LARGE_INTEGER countsPerSec = getFrequency();
	double secondsPerCount = 1.0 / (double)countsPerSec.QuadPart;
	// The physics clock is just used to run the physics and runs asynchronously with the gameclock
	LARGE_INTEGER currTimeStamp = getTimeStamp();
	LARGE_INTEGER prevTimeStamp = currTimeStamp;
	// There's an inner loop in here where things happen once every TickMs. These variables are for that.
	LARGE_INTEGER gameClockTimeOffsetStamp = getTimeStamp();
	double gameClockTimeOffset = (double)gameClockTimeOffsetStamp.QuadPart * secondsPerCount;
	const unsigned int gameTickMs = 16;
	double gameTickS = (double)gameTickMs / 1000.0;
	// Absolute start
	double timeStart = (double)getTimeStamp().QuadPart * secondsPerCount;



	// Bullet physics initialization
	// Broadphase object
	btBroadphaseInterface* broadphase = new btDbvtBroadphase();
	// Collision dispatcher with default config
	btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);
	// Register collision algorithm (needed for mesh collisions)
	// btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
	// Register physics solver
	// (Single threaded)
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;
	// ==================================
	// Create the physics world
	// ==================================
	btDiscreteDynamicsWorld* dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
	dynamicsWorld->setGravity(btVector3(0, -9.82, 0));



	// Artemis
	// Create and initialize systems
	artemis::SystemManager * sysManager = m_world.getSystemManager();
	//MovementSystem * movementsys = (MovementSystem*)sm->setSystem(new MovementSystem());
	//addGameLogic(movementsys);
	m_rigidBodySystem = (RigidBodySystem*)sysManager->setSystem(new RigidBodySystem(dynamicsWorld));
	m_renderSystem = (RenderSystem*)sysManager->setSystem(new RenderSystem(m_graphicsDevice));
	m_controllerSystem = (ControllerSystem*)sysManager->setSystem(new ControllerSystem());
	sysManager->initializeAll();




	// Combine Physics with our stuff!
	PhysicsWorldHandler physicsWorldHandler(dynamicsWorld,m_controllerSystem);




	// Entity manager fetch
	artemis::EntityManager * entityManager = m_world.getEntityManager();

	// Create a box entity
	for (int i = 0; i < 100;i++)
	{
		artemis::Entity & box = entityManager->create();
		glm::vec3 pos = glm::vec3(20.0f*sin(i*0.1f), 10.0f + i*4.0f, 20.0f*cos(i*0.1f));
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
		box.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(0.5f, 0.5f, 0.5f)), 1.0f));
		box.addComponent(new RenderComponent());
		box.addComponent(new TransformComponent(pos,
			glm::quat(glm::vec3(TWOPI*0.05f, 0.0f, 0.0f))
			));
		box.refresh();
	}

	artemis::Entity & box = entityManager->create();
	glm::vec3 pos = glm::vec3(0.0f);
	RigidBodyComponent* rb = new RigidBodyComponent(new btBoxShape(btVector3(0.5f, 0.5f, 0.5f)), 1.0f);
	box.addComponent(rb);
	box.addComponent(new RenderComponent());
	box.addComponent(new TransformComponent(pos,
		glm::quat(glm::vec3(TWOPI*0.05f, 0.0f, 0.0f))
		));
	box.refresh();

	// Create a ground entity

	artemis::Entity & ground = entityManager->create();
	ground.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(50.0f, 10.0f, 50.0f)), 0.0f));
	ground.addComponent(new RenderComponent());
	ground.addComponent(new TransformComponent(glm::vec3(0.0f, -20.0f, 0.0f), 
		glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)),
		glm::vec3(100.0f, 20.0f, 100.0f)));
	ground.refresh();


	// Create axes
	artemis::Entity & axisC = entityManager->create();
	axisC.addComponent(new RenderComponent());
	axisC.addComponent(new TransformComponent(glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(2.0f, 2.0f, 2.0f)));
	axisC.refresh();
	//
	artemis::Entity & axisX = entityManager->create();
	axisX.addComponent(new RenderComponent());
	axisX.addComponent(new TransformComponent(glm::vec3(5.0f, 0.0f, 0.0f),
		glm::vec3(10.0f, 1.0f, 1.0f)));
	axisX.refresh();
	//
	artemis::Entity & axisY = entityManager->create();
	axisY.addComponent(new RenderComponent());
	axisY.addComponent(new TransformComponent(glm::vec3(0.0f, 10.0f, 0.0f),
		glm::vec3(1.0f, 20.0f, 1.0f)));
	axisY.refresh();
	//
	artemis::Entity & axisZ = entityManager->create();
	axisZ.addComponent(new RenderComponent());
	axisZ.addComponent(new TransformComponent(glm::vec3(0.0f, 0.0f, 5.0f),
		glm::vec3(1.0f, 2.0f, 10.0f)));
	axisZ.refresh();

	// Linked boxes
	{
		artemis::Entity & parentJoint = entityManager->create();
		glm::vec3 pos = glm::vec3(glm::vec3(0.0f, 20.0f, 0.0f));
		//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
		parentJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(1.0f, 2.0f, 1.0f)), 0.0f));
		parentJoint.addComponent(new RenderComponent());
		parentJoint.addComponent(new TransformComponent(pos,
			glm::quat(glm::vec3(PI*0.1f, 0.0f, 0.0f)),
			glm::vec3(2.0f, 4.0f, 2.0f)));
		parentJoint.refresh();
		//
		artemis::Entity* prev = &parentJoint;
		artemis::Entity* mid = NULL;
		for (int i = 0; i < 50; i++)
		{
			artemis::Entity & childJoint = entityManager->create();
			pos = glm::vec3(glm::vec3(0.0f, 0.0f, 20.0f));
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
			childJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(0.5f, 2.0f, 0.5f)), 1.0f));
			childJoint.addComponent(new RenderComponent());
			childJoint.addComponent(new TransformComponent(pos, glm::vec3(1.0f, 4.0f, 1.0f)));
			ConstraintComponent::ConstraintDesc constraintDesc{ glm::vec3(0.0f, 2.0f, 0.0f),	  // child
				glm::vec3(0.0f, -2.0f, 0.0f), // parent
				{ glm::vec3(-HALFPI, 0.0f, 0.0f), glm::vec3(HALFPI, 0.0f, 0.0f) },
				false };
			childJoint.addComponent(new ConstraintComponent(prev, constraintDesc));
			childJoint.refresh();
			prev = &childJoint;
			if (i == 25)
				mid = &childJoint;
		}
		//
		prev = mid;
		for (int i = 0; i < 25; i++)
		{
			artemis::Entity & childJoint = entityManager->create();
			pos = glm::vec3(glm::vec3(0.0f, 0.0f, 20.0f));
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
			childJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(0.5f, 2.0f, 0.5f)), 1.0f));
			childJoint.addComponent(new RenderComponent());
			childJoint.addComponent(new TransformComponent(pos, glm::vec3(1.0f, 4.0f, 1.0f)));
			ConstraintComponent::ConstraintDesc constraintDesc{ glm::vec3(0.0f, 2.0f, 0.0f),	  // child
				glm::vec3(0.0f, -2.0f, 0.0f), // parent
				{ glm::vec3(-HALFPI, 0.0f, 0.0f), glm::vec3(HALFPI, 0.0f, 0.0f) },
				false };
			childJoint.addComponent(new ConstraintComponent(prev, constraintDesc));
			childJoint.refresh();
			prev = &childJoint;
		}
	}

	// Test of controller
	{
		artemis::Entity & parentJoint = entityManager->create();
		glm::vec3 pos = glm::vec3(glm::vec3(0.0f, 10.0f, -10.0f));
		//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
		parentJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(1.0f, 2.0f, 1.0f)), 0.0f));
		parentJoint.addComponent(new RenderComponent());
		parentJoint.addComponent(new TransformComponent(pos,
			glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)),
			glm::vec3(2.0f, 4.0f, 2.0f)));
		parentJoint.refresh();
		//
		artemis::Entity* prev = &parentJoint;
		artemis::Entity* mid = NULL;
		for (int i = 0; i < 2; i++)
		{
			artemis::Entity & childJoint = entityManager->create();
			pos = glm::vec3(glm::vec3(0.0f, 0.0f, 20.0f));
			//(float(i) - 50, 10.0f+float(i)*4.0f, float(i)*0.2f-50.0f);
			childJoint.addComponent(new RigidBodyComponent(new btBoxShape(btVector3(0.5f, 2.0f, 0.5f)), 1.0f));
			childJoint.addComponent(new RenderComponent());
			childJoint.addComponent(new TransformComponent(pos, glm::vec3(1.0f, 4.0f, 1.0f)));
			ConstraintComponent::ConstraintDesc constraintDesc{ glm::vec3(0.0f, 2.0f, 0.0f),	  // child
				glm::vec3(0.0f, -2.0f, 0.0f), // parent
				{ glm::vec3(-HALFPI, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f) },
				false };
			childJoint.addComponent(new ConstraintComponent(prev, constraintDesc));
			childJoint.refresh();
			prev = &childJoint;
		}
	}


	MeasurementBin<string> measurer;
	measurer.activate();

	// Message pump struct
	MSG msg = {0};

	// secondary run variable
	// lets non-context systems quit the program
	bool run=true;

	// Dry run, so artemis have run before physics first step
	gameUpdate(0.0f);

	while (!m_context->closeRequested() && run)
	{
		if (!pumpMessage(msg))
		{
			// Start by rendering
			render();

			double time = (double)getTimeStamp().QuadPart*secondsPerCount - timeStart;
			//if (time>19.0)
			//{
			//	if (rb->isInited())
			//	{
			//		btTransform btt;
			//		rb->getRigidBody()->getMotionState()->getWorldTransform(btt);
			//		measurer.saveMeasurement(string("x: ") + toString(btt.getOrigin().x()) + ",y: " + toString(btt.getOrigin().y()) + ",z: " + toString(btt.getOrigin().z()),
			//			time);
			//	}
			//}

			// Physics handling part of the loop
			// ========================================================
			/* This, like the rendering, ticks every time around.
			Bullet does the interpolation for us. */
			currTimeStamp = getTimeStamp();
			double phys_dt = (double)m_timeScale*(double)(currTimeStamp.QuadPart - prevTimeStamp.QuadPart) * secondsPerCount;


			//if (rb->isInited())
			//	rb->getRigidBody()->applyForce(btVector3(0.0f, 20.0f, 0.0f), btVector3(0.0f, 0.0f, 0.0f));

			// Tick the bullet world. Keep in mind that bullet takes seconds
			dynamicsWorld->stepSimulation((btScalar)phys_dt, 10);
			// ========================================================



			prevTimeStamp = currTimeStamp;

			//if (time>20.0) run = false;

			// Game Clock part of the loop
			// ========================================================
			double dt = ((double)getTimeStamp().QuadPart*secondsPerCount - gameClockTimeOffset);
			// Game clock based updates
			while (dt >= gameTickS)
			{
				dt -= gameTickS;
				gameClockTimeOffset += gameTickS;
				// Handle all input

				{
					processInput();
					// Update logic
					double interval = gameTickS;

					handleContext(interval, phys_dt);
					gameUpdate(interval);
				}
			}
			// ========================================================
		}

	}

	//#ifdef _DEBUG
	//	measurer.saveResults("../output/determinismTest_Debug");
	//#else
	//	measurer.saveResults("../output/determinismTest_Release");
	//#endif



	entityManager->removeAllEntities();

}


LARGE_INTEGER App::getTimeStamp()
{
	LARGE_INTEGER stamp;
	QueryPerformanceCounter(&stamp);
	return stamp;
}

LARGE_INTEGER App::getFrequency()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
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

bool App::pumpMessage( MSG& p_msg )
{
	bool res = PeekMessage(&p_msg, NULL, 0, 0, PM_REMOVE)>0?true:false;
	if (res)
	{
		TranslateMessage(&p_msg);
		DispatchMessage(&p_msg);
	}
	return res;
}


void App::processInput()
{
	m_input->run();
}

void App::handleContext(double p_dt, double p_physDt)
{
	// apply resizing on graphics device if it has been triggered by the context
	if (m_context->isSizeDirty())
	{
		pair<int, int> sz = m_context->getSize();
		m_graphicsDevice->updateResolution(sz.first, sz.second);
	}
	// Print fps in window head border
	m_fpsUpdateTick -= (float)p_dt;
	if (m_fpsUpdateTick <= 0.0f)
	{
		float fps = (1.0f / (float)(p_dt*1000.0f))*1000.0f;
		float pfps = 1.0f / (float)p_physDt;
		m_context->updateTitle((" | Game FPS: " + toString(fps) + " | Phys FPS: " + toString(pfps)).c_str());
		m_fpsUpdateTick = 0.3f;
	}
}

void App::gameUpdate( double p_dt )
{
	float dt = (float)p_dt;
	float game_dt = m_timeScale*(float)p_dt;
	// temp controller update code
	updateController(dt);
	m_controller->setFovFromAngle(52.0f, m_graphicsDevice->getAspectRatio());
	m_controller->update(dt);
	// Get camera info to buffer
	std::memcpy(&m_vp->accessBuffer, &m_controller->getViewProjMatrix(), sizeof(float)* 4 * 4);
	m_vp->update();

	if (m_timePauseStepToggle && m_timeScale > 0.0f)
		m_timeScale = 0.0f;
	if (m_input->g_kb->isKeyDown(KC_RETURN))
	{
		if (!m_timeScaleToggle)
		{
			if (m_timeScale < 1.0f)
				m_timeScale = 1.0f;
			else
				m_timeScale = 0.0f;
			m_timeScaleToggle = true;
		}
	}
	else
	{
		m_timeScaleToggle = false;
	}
	if (m_input->g_kb->isKeyDown(KC_NUMPAD6))
	{
		if (m_timeScale < 1.0f && !m_timePauseStepToggle)
		{
			m_timePauseStepToggle = true;
			m_timeScale = 1.0f;
		}
	}
	else
	{
		m_timePauseStepToggle = false;
	}


	//for (unsigned int i = 1; i < m_instances->getElementCount();i++)
	//{
	//	glm::mat4* firstTransform = m_instances->readElementPtrAt(i);
	//	*firstTransform = glm::transpose(*firstTransform);
	//	*firstTransform *= glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, sin(i)+cos(i), 0.0f)*0.1f*(float)p_dt);
	//	*firstTransform = glm::transpose(*firstTransform);
	//	//m_instances->writeElementAt(i, firstTransform);
	//	//m_instances[0].accessBuffer = firstTransform;
	//	int x = 0;
	//}
	//m_instances->update();

	// Update entity systems
	m_world.loopStart();
	m_world.setDelta(game_dt);
	// Physics result gathering have to run first
	m_rigidBodySystem->executeDeferredConstraintInits();
	m_rigidBodySystem->process();
	m_controllerSystem->buildCheck();
	// Run all other systems, for which order doesn't matter
	processSystemCollection(&m_orderIndependentSystems);
	// Render system is processed last
	m_renderSystem->process();


	// Run the devices
	// ---------------------------------------------------------------------------------------------

	//int v[11] = {'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};
	//
	//// Serial (CPU)
	//
	//
	//// PPL (CPU)
	//int pplRes[11];
	//concurrency::parallel_for(0, 11, [&](int n) {
	//	pplRes[n]=v[n]+1;
	//});
	//for(unsigned int i = 0; i < 11; i++) 
	//	std::cout << static_cast<char>(pplRes[i]); 
	//
	//// C++AMP (GPU)
	//concurrency::array_view<int> av(11, v); 
	//concurrency::parallel_for_each(av.extent, [=](concurrency::index<1> idx) restrict(amp) 
	//{ 
	//	av[idx] += 1; 
	//});
	//
	//
	//// Print C++AMP
	//for (unsigned int i = 0; i < 11; i++)
	//{
	//	char ch = static_cast<char>(av[i]);
	//	DEBUGPRINT(( toString(ch).c_str() ));
	//}
	//DEBUGPRINT((string("\n").c_str()));
}


void App::render()
{
	// Clear render targets
	m_graphicsDevice->clearRenderTargets();
	// Run passes
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_WIREFRAMEPASS, m_vp, m_renderSystem->getInstances());
	// Flip!
	m_graphicsDevice->flipBackBuffer();										
}

// Add a system for game logic processing
void App::addOrderIndependentSystem(artemis::EntityProcessingSystem* p_system)
{
	m_orderIndependentSystems.push_back(p_system);
}


void App::processSystemCollection(vector<artemis::EntityProcessingSystem*>* p_systemCollection)
{
	unsigned int count = (unsigned int)p_systemCollection->size();
	for (unsigned int i = 0; i < count; i++)
	{
		artemis::EntityProcessingSystem* system = (*p_systemCollection)[i];
		system->process();
	}
}
