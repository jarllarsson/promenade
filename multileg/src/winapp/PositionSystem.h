#pragma once

#include <Artemis.h>
#include <DebugPrint.h>
/*
class MovementSystem : public artemis::EntityProcessingSystem {
private:
	//artemis::ComponentMapper<MovementComponent> velocityMapper;
	artemis::ComponentMapper<PositionComponent> positionMapper;

public:
	MovementSystem() {
		//addComponentType<VelocityComponent>();
		addComponentType<PositionComponent>();
	};

	virtual void initialize() {
		//velocityMapper.init(*world);
		positionMapper.init(*world);
	};

	virtual void processEntity(artemis::Entity &e) {
		positionMapper.get(e)->posX += 1.0f;
			//velocityMapper.get(e)->velocityX * world->getDelta();
		positionMapper.get(e)->posY += 1.0f;
			//velocityMapper.get(e)->velocityY * world->getDelta();
		DEBUGPRINT(( (string("ART")+toString(positionMapper.get(e)->posX)).c_str()));
	};

};*/