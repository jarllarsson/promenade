#include "RigidBodySystem.h"
#include <ToString.h>


void RigidBodySystem::removed(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	ConstraintComponent* constraint = (ConstraintComponent*)e.getComponent<ConstraintComponent>();
	if (constraint)
		constraint->forceRemove(m_dynamicsWorldPtr);
	//if (rigidBody)
	//{
	//	if (rigidBody->isInited())
	//	{
	//		//checkForConstraintsToRemove(e, rigidBody);
	//		m_dynamicsWorldPtr->removeRigidBody(rigidBody->getRigidBody());
	//	}
	//}
};

void RigidBodySystem::added(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	TransformComponent* transform = transformMapper.get(e);
	if (!rigidBody->isInited())
	{
		// Set up the rigidbody
		btTransform t;
		// Construct matrix without scale (or the shape will bug)
		glm::mat4 translate = glm::translate(glm::mat4(1.0f), transform->getPosition());
		glm::mat4 rotate = glm::mat4_cast(transform->getRotation());
		glm::mat4 mat = translate * rotate;
		// Read to bt matrix
		t.setFromOpenGLMatrix(glm::value_ptr(mat));
		// Init motionstate with matrix
		btDefaultMotionState* motionState = new btDefaultMotionState(t);
		// Calculate inertia, using our collision shape
		btVector3 inertia(0, 0, 0);
		float mass = rigidBody->getMass();
		btCollisionShape* collisionShape = rigidBody->getCollisionShape();
		collisionShape->calculateLocalInertia(mass, inertia);
		// Construction info
		btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, motionState, collisionShape, inertia);
		btRigidBody* rigidBodyInstance = new btRigidBody(rigidBodyCI);
		rigidBodyInstance->setDamping(0.0f, 0.1f);
		//float test = rigidBodyInstance->getFriction();
		rigidBodyInstance->setFriction(0.8f); // custom friction, default is 0.5
		rigidBodyInstance->setActivationState(DISABLE_DEACTIVATION);
		rigidBodyInstance->setCcdMotionThreshold(10);
		btVector3 c; float r;
		collisionShape->getBoundingSphere(c, r);
		rigidBodyInstance->setCcdSweptSphereRadius(r*0.2f);
		// If collision registration is activated,
		// we need to prepare a callback
		if (rigidBody->isRegisteringCollisions())
		{
			OnCollisionCallback* callback = new OnCollisionCallback(rigidBodyInstance, rigidBody);
			rigidBody->addCollisionCallback(callback); // the component handles dealloc upon destruction
		}
		
		// Add rigidbody to list
		unsigned int uid = m_rigidBodyEntities.add(&e);
		//
		rigidBody->init(uid, rigidBodyInstance, m_dynamicsWorldPtr);
		m_dynamicsWorldPtr->addRigidBody(rigidBody->getRigidBody(),rigidBody->m_collisionLayerType,rigidBody->m_collidesWithLayer);
		// check if entity has constraints, if so, and if they're uninited, add to
		// list for batch init (as they must have both this entity's and parent's rb in phys world.
		checkForNewConstraints(e);
	}
};


void RigidBodySystem::processEntity(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	TransformComponent* transform = transformMapper.get(e);
	// Update transform of object
	if (rigidBody->isInited())
	{
		btRigidBody* body = rigidBody->getRigidBody();
		if (body != NULL/* && body->isInWorld() && body->isActive()*/)
		{
			btMotionState* motionState = body->getMotionState();
			btTransform physTransform;
			motionState->getWorldTransform(physTransform);
			// Store old position
			glm::vec3 oldpos = transform->getPosition();
			// update the transform component
			// Get the transform from Bullet and into mat
			glm::mat4 mat(0.0f);
			// first we need to keep scale as bullet doesn't
			glm::vec3 scale = transform->getScale();
			physTransform.getOpenGLMatrix(glm::value_ptr(mat));
			transform->setMatrix(mat);
			transform->setScaleToMatrix(scale);
			// Now if the calculation of velocity is enabled, sample and calculate it
			if (m_measureVelocityAndAcceleration)
			{
				glm::vec3 npos = transform->getPosition(); // new pos
				// get old velocity
				glm::vec3 oldvel = rigidBody->getVelocity();
				// velocity is delta position
				rigidBody->setVelocityStat(npos - oldpos);
				glm::vec3 nvel = rigidBody->getVelocity();
				// acceleration is delta velocity
				rigidBody->setAccelerationStat(nvel - oldvel);
			}
// 			if (rigidBody->isRegisteringCollisions())
// 			{
// 				m_dynamicsWorldPtr->contactTest(body, *rigidBody->getCollisionCallbackFunc());
// 				if (rigidBody->isColliding())
// 				{
// 					glm::vec3 hitPos = transform->getPosition() + rigidBody->getCollisionPoint();
// 					dbgDrawer()->drawLine(hitPos, hitPos + glm::vec3(0.0f, 1.0f, 0.0f), dawnBringerPalRGB[COL_ORANGE], dawnBringerPalRGB[COL_RED]);
// 				}
// 			}
// 			// Extra debug
			if (rigidBody->isColliding())
			{
				glm::vec3 hitPos = /*MathHelp::transformPosition(transform->getMatrixPosRot(),*/rigidBody->getCollisionPoint()/*)*/;
				dbgDrawer()->drawLine(hitPos - glm::vec3(0.0f, 0.5f, 0.0f), hitPos + glm::vec3(0.0f, 0.5f, 0.0f), dawnBringerPalRGB[COL_RED], dawnBringerPalRGB[COL_RED]);
				dbgDrawer()->drawLine(hitPos - glm::vec3(0.5f, 0.0f, 0.0f), hitPos + glm::vec3(0.5f, 0.0f, 0.0f), dawnBringerPalRGB[COL_RED], dawnBringerPalRGB[COL_RED]);
				dbgDrawer()->drawLine(hitPos - glm::vec3(0.0f, 0.0f, 0.5f), hitPos + glm::vec3(0.0f, 0.0f, 0.5f), dawnBringerPalRGB[COL_RED], dawnBringerPalRGB[COL_RED]);
			}
			//
			if (m_stateDbgRecorder != NULL && m_stateDbgRecorder->isActive())
			{
				btTransform* btt = &physTransform;
				m_stateString += string("\n,") +ToString(e.getUniqueId())+string(" x: ") + ToString(btt->getOrigin().x()) + ",y: " + ToString(btt->getOrigin().y()) + ",z: " + ToString(btt->getOrigin().z());
			}
		}
	}
}

void RigidBodySystem::checkForNewConstraints(artemis::Entity &e)
{
	ConstraintComponent* constraint = (ConstraintComponent*)e.getComponent<ConstraintComponent>();
	if (constraint != NULL && !constraint->isInited())
	{
		m_constraintCreationsList.push_back(&e);
	}
}

// void RigidBodySystem::checkForConstraintsToRemove(artemis::Entity &e, RigidBodyComponent* p_rigidBody)
// {
// 	// First check if we have the constraint (we are child)
// 	ConstraintComponent* constraint = (ConstraintComponent*)e.getComponent<ConstraintComponent>();
// 	// Otherwise, if this is the parent, fetch the stowaway ptr for this case:
// 	if (constraint == NULL) constraint = p_rigidBody->getChildConstraint();
// 	if (constraint != NULL && constraint->isInited() && !constraint->isRemoved())
// 	{
// 		constraint->forceRemove(m_dynamicsWorldPtr);
// 	}
// }

void RigidBodySystem::executeDeferredConstraintInits()
{
	for (unsigned int i = 0; i < m_constraintCreationsList.size(); i++)
	{
		setupConstraints(m_constraintCreationsList[i]);
	}
	m_constraintCreationsList.clear();
}

void RigidBodySystem::setupConstraints(artemis::Entity *e)
{
	// This (child) is guaranteed to have the right components
	// As it was already processed
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(*e);
	TransformComponent* transform = transformMapper.get(*e);
	ConstraintComponent* constraint = (ConstraintComponent*)e->getComponent<ConstraintComponent>();
	if (constraint != NULL && !constraint->isInited() && constraint->getParent() != NULL)
	{
		RigidBodyComponent* parentRigidBody = (RigidBodyComponent*)constraint->getParent()->getComponent<RigidBodyComponent>();
		TransformComponent* parentTransform = (TransformComponent*)constraint->getParent()->getComponent<TransformComponent>();
		if (parentRigidBody != NULL && parentTransform != NULL && parentRigidBody->isInited())
		{
			btRigidBody* rigidBodyInstance = rigidBody->getRigidBody();
			btRigidBody* parentRigidBodyInstance = parentRigidBody->getRigidBody();
			ConstraintComponent::ConstraintDesc constraintdesc = *constraint->getDesc();
			// create a universal joint using generic 6DOF constraint
			// add some (arbitrary) data to build constraint frames
			glm::vec3 gparentAnchor = constraintdesc.m_parentLocalAnchor;
			glm::vec3 gchildAnchor = constraintdesc.m_localAnchor;
			// put in btvectors
			btVector3 parentAnchor(gparentAnchor.x, gparentAnchor.y, gparentAnchor.z);
			btVector3 childAnchor(gchildAnchor.x, gchildAnchor.y, gchildAnchor.z);
			// Get limits
			btVector3 angularLimLow(constraintdesc.m_angularDOF_LULimits[0].x, constraintdesc.m_angularDOF_LULimits[0].y, constraintdesc.m_angularDOF_LULimits[0].z);
			btVector3 angularLimHigh(constraintdesc.m_angularDOF_LULimits[1].x, constraintdesc.m_angularDOF_LULimits[1].y, constraintdesc.m_angularDOF_LULimits[1].z);
			// Linear lims read here if used!
			//
			// build frame basis
			// 6DOF constraint uses Euler angles and to define limits
			// X - allowed limits are (-PI,PI);
			// Y - second (allowed limits are (-PI/2 + epsilon, PI/2 - epsilon), where epsilon is a small positive number 
			// Z - allowed limits are (-PI,PI);
			btTransform frameInParent, frameInChild;
			//frameInParent.setFromOpenGLMatrix(glm::value_ptr(parentTransform->getMatrixPosRot()));
			frameInParent = btTransform::getIdentity();
			frameInParent.setOrigin(parentAnchor);
			//frameInChild.setFromOpenGLMatrix(glm::value_ptr(transform->getMatrixPosRot()));
			frameInChild = btTransform::getIdentity();
			frameInChild.setOrigin(childAnchor);
			//
			parentRigidBodyInstance->setActivationState(DISABLE_DEACTIVATION);
			rigidBodyInstance->setActivationState(DISABLE_DEACTIVATION);
			// now create the constraint
			btGeneric6DofConstraint* pGen6DOF = new btGeneric6DofConstraint(*parentRigidBodyInstance, *rigidBodyInstance, frameInParent, frameInChild, true);
			// linear limits in our case are allowed offset of origin of frameInB in frameInA, so set them to zero
			pGen6DOF->setLinearLowerLimit(btVector3(0., 0., 0.));
			pGen6DOF->setLinearUpperLimit(btVector3(0., 0., 0.));
			// set limits for parent (axis z) and child (axis Y)
			pGen6DOF->setAngularLowerLimit(angularLimLow);
			pGen6DOF->setAngularUpperLimit(angularLimHigh);
			// add constraint to world
			m_dynamicsWorldPtr->addConstraint(pGen6DOF, !constraintdesc.m_collisionBetweenLinked);
			constraint->init(pGen6DOF, e);
			parentRigidBody->addChildConstraint(constraint);
		}
	}
}

void RigidBodySystem::lateUpdate()
{
	if (m_stateDbgRecorder!=NULL && m_stateDbgRecorder->isActive())
	{
		m_stateDbgRecorder->saveMeasurementRelTStamp(m_stateString+"\n", world->getDelta());
		m_stateString.clear();
	}
}

void RigidBodySystem::fixedUpdate(float p_dt)
{
	for (unsigned int i = 0; i < m_rigidBodyEntities.getSize(); i++)
	{
		artemis::Entity* e = m_rigidBodyEntities[i];
		RigidBodyComponent* rigidBody = rigidBodyMapper.get(*e);
		TransformComponent* transform = transformMapper.get(*e);
		if (rigidBody->isInited())
		{
			if (rigidBody->isRegisteringCollisions())
			{
				btRigidBody* body = rigidBody->getRigidBody();
				rigidBody->setCollidingStat(false);
				m_dynamicsWorldPtr->contactTest(body, *rigidBody->getCollisionCallbackFunc());
			}
		}
	}
}

