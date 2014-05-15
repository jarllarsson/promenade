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

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>


#include "PositionSystem.h"


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
	m_controller = new TempController();
	m_input = new Input();
	m_input->doStartup(m_context->getWindowHandle());
	//
	for (int x = 0; x < 100; x++)
	for (int y = 0; y < 2; y++)
	for (int z = 0; z < 100; z++)
	{
		glm::mat4 transMat = glm::translate(glm::mat4(1.0f),
			glm::vec3((float)x*2.0f -100.0f, (float)y*2.0f-100.0f, (float)z*2.0f));
		transMat = glm::transpose(transMat);
		m_instance.push_back(transMat);
	}
	m_instances = m_graphicsDevice->getBufferFactoryRef()->createMat4InstanceBuffer((void*)&m_instance[0], (unsigned int)m_instance.size());
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
	LARGE_INTEGER countsPerSec = getFrequency();
	double secondsPerCount = 1.0 / (double)countsPerSec.QuadPart;
	// The physics clock is just used to run the physics and runs asynchronously with the gameclock
	LARGE_INTEGER currTimeStamp = getTimeStamp();
	LARGE_INTEGER prevTimeStamp = currTimeStamp;
	// There's an inner loop in here where things happen once every TickMs. These variables are for that.
	LARGE_INTEGER timeStartStamp = getTimeStamp();
	double timeStart = (double)timeStartStamp.QuadPart * secondsPerCount;
	const unsigned int gameTickMs = 16;
	double gameTickS = (double)gameTickMs / 1000.0;


	// Artemis
	// Create and initialize systems
	artemis::SystemManager * sm = m_world.getSystemManager();
	//MovementSystem * movementsys = (MovementSystem*)sm->setSystem(new MovementSystem());
	//addGameLogic(movementsys);
	sm->initializeAll();

	// Create an entity
	artemis::EntityManager * em = m_world.getEntityManager();
	artemis::Entity & player = em->create();
	//player.addComponent(new MovementComponent(2, 4));
	//player.addComponent(new PositionComponent(0, 0));
	player.refresh();

	//PositionComponent * comp = (PositionComponent*)player.getComponent<PositionComponent>();





	// BP TEST
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
	// -----------
	// Objects
	// -----------
	// Create shapes
	btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);
	btCollisionShape* fallShape = new btSphereShape(1);
	// Create motion state for ground
	// http://bulletphysics.org/mediawiki-1.5.8/index.php/MotionStates#MotionStates
	// Are used to retreive the calculated transform data from bullet
	btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
		btVector3(0, -100, 0)));
	// Create rigidbody for ground
	// Bullet considers passing a mass of zero equivalent to making a body with infinite mass - it is immovable.
	btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));
	btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
	// Add ground to world
	dynamicsWorld->addRigidBody(groundRigidBody);

	// Same procedure for sphere
	// Create rigidbody for sphere (with motion state 50m above ground)
	btDefaultMotionState* fallMotionState =
		new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 50, 0)));
	btScalar mass = 1; // 1kg
	btVector3 fallInertia(0, 0, 0);
	fallShape->calculateLocalInertia(mass, fallInertia); // sphere inertia
	// And the rigidbody
	btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI(mass, fallMotionState, fallShape, fallInertia);
	btRigidBody* fallRigidBody = new btRigidBody(fallRigidBodyCI);
	dynamicsWorld->addRigidBody(fallRigidBody);













	// Message pump struct
	MSG msg = {0};

	// secondary run variable
	// lets non-context systems quit the program
	bool run=true;

	while (!m_context->closeRequested() && run)
	{
		if (!pumpMessage(msg))
		{

			render();

			// Physics handling part of the loop
			/* This, like the rendering, ticks every time around.
			Bullet does the interpolation for us. */

			currTimeStamp = getTimeStamp();
			//time_physics_curr = getMilliseconds();
			double phys_dt = (double)(currTimeStamp.QuadPart - prevTimeStamp.QuadPart) * secondsPerCount;

			// Tick the bullet world. Keep in mind that bullet takes seconds
			physUpdate(phys_dt);


			dynamicsWorld->stepSimulation((btScalar)phys_dt, 10);
			btTransform trans;
			fallRigidBody->getMotionState()->getWorldTransform(trans);
			DEBUGPRINT(( (toString(trans.getOrigin().getY())+"\n").c_str() ));

			glm::mat4* firstTransform = m_instances->readElementPtrAt(0);
			*firstTransform = glm::transpose(*firstTransform);
			*firstTransform = glm::translate(glm::mat4(1.0f), glm::vec3(trans.getOrigin().getX(), trans.getOrigin().getY(), trans.getOrigin().getZ()));
			*firstTransform = glm::transpose(*firstTransform);

			m_instances->update();


			prevTimeStamp = currTimeStamp;


			// Game Clock part of the loop
			/*  This ticks once every TickMs milliseconds on average */
			double dt = (double)getTimeStamp().QuadPart*secondsPerCount - timeStart;

			// Game clock based updates
			while (dt >= gameTickS)
			{
				dt -= gameTickS;
				timeStart += gameTickS;
				// Handle all input

				{
					processInput();
					// Update logic
					double interval = gameTickS;
					handleContext(interval, phys_dt);
					gameUpdate(interval);
				}
			}
		}

	}
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
	// temp controller update code
	updateController(dt);
	m_controller->setFovFromAngle(52.0f, m_graphicsDevice->getAspectRatio());
	m_controller->update(dt);
	// Get camera info to buffer
	std::memcpy(&m_vp->accessBuffer, &m_controller->getViewProjMatrix(), sizeof(float)* 4 * 4);
	m_vp->update();

	for (unsigned int i = 1; i < m_instances->getElementCount();i++)
	{
		glm::mat4* firstTransform = m_instances->readElementPtrAt(i);
		*firstTransform = glm::transpose(*firstTransform);
		*firstTransform *= glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, sin(i)+cos(i), 0.0f)*0.1f*(float)p_dt);
		*firstTransform = glm::transpose(*firstTransform);
		//m_instances->writeElementAt(i, firstTransform);
		//m_instances[0].accessBuffer = firstTransform;
		int x = 0;
	}
	m_instances->update();

	// Update entity systems
	m_world.loopStart();
	m_world.setDelta(p_dt);
	processSystemCollection(&m_gameLogicSystems);


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

void App::physUpdate(double p_dt)
{
	//mWorld->stepSimulation((float)p_dt, 10);
}

void App::render()
{
	// Clear render targets
	m_graphicsDevice->clearRenderTargets();
	// Run passes
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);
	m_graphicsDevice->executeRenderPass(GraphicsDevice::P_WIREFRAMEPASS, m_vp, m_instances);
	// Flip!
	m_graphicsDevice->flipBackBuffer();										
}

// Add a system for game logic processing
void App::addGameLogic(artemis::EntityProcessingSystem* p_system)
{
	m_gameLogicSystems.push_back(p_system);
}

// Add system for processing in the physics ticker
void App::addPhysicsLogic(artemis::EntityProcessingSystem* p_system)
{
	m_physicsLogicSystems.push_back(p_system);
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
