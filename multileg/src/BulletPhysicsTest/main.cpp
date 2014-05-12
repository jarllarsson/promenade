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
																				   btVector3(0, -1, 0)));
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


	// ==================================
	// ==================================
	//               MAIN
	// ==================================
	// ==================================

	// Simulate for 300 iterations
	for (int i = 0; i<300; i++) 
	{
		dynamicsWorld->stepSimulation(1 / 60.f, 10); // 60hz

		btTransform trans;
		fallRigidBody->getMotionState()->getWorldTransform(trans);

		std::cout << "sphere height: " << trans.getOrigin().getY() << std::endl;
	}


	// ==================================
	// ==================================

	// Bullet has a policy of "whoever allocates, also deletes" memory
	dynamicsWorld->removeRigidBody(fallRigidBody);
	delete fallRigidBody->getMotionState();
	delete fallRigidBody;

	dynamicsWorld->removeRigidBody(groundRigidBody);
	delete groundRigidBody->getMotionState();
	delete groundRigidBody;

	delete groundShape;
	delete fallShape;

	delete dynamicsWorld;
	delete solver;
	delete dispatcher;
	delete collisionConfiguration;
	delete broadphase;



	return 0;
}