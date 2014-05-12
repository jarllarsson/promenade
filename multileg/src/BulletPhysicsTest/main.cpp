#include <iostream>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>
#include <vld.h> 

int main()
{
	std::cout << "Hello World!" << std::endl;

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
	dynamicsWorld->setGravity(btVector3(0, -10, 0));


	// ==================================
	// ==================================
	//               MAIN
	// ==================================
	// ==================================




	// ==================================
	// ==================================

	// Bullet has a policy of "whoever allocates, also deletes" memory
	delete dynamicsWorld;
	delete solver;
	delete dispatcher;
	delete collisionConfiguration;
	delete broadphase;

	return 0;
}