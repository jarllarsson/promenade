#pragma once
#include <Artemis.h>
#include "Toolbar.h"
#include "DebugDrawBatch.h"

// =======================================================================================
//                              AdvancedEntitySystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Extended ES that allows for further specialized access to the game environment
///        
/// # AdvancedEntitySystem
/// 
/// 18-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class AdvancedEntitySystem : public artemis::EntityProcessingSystem
{
public:
	AdvancedEntitySystem() {}
	virtual ~AdvancedEntitySystem() {}

	static void registerDebugToolbar(Toolbar* p_toolbar){ m_toolbar = p_toolbar; }
	static void registerDebugDrawBatch(DebugDrawBatch* p_dbgDrawer){ m_dbgDrawer = p_dbgDrawer; }

	virtual void fixedUpdate(float p_dt) {}
protected:
	static Toolbar* dbgToolbar() { return m_toolbar; }
	static DebugDrawBatch* dbgDrawer() { return m_dbgDrawer; }
private:	
	static Toolbar* m_toolbar;
	static DebugDrawBatch* m_dbgDrawer;
};