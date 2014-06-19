#pragma once
#include <Artemis.h>
#include "Toolbar.h"
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


protected:
	static Toolbar* dbgToolbar() { return m_toolbar; }

private:	
	static Toolbar* m_toolbar;
};