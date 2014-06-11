#pragma once
#include <Util.h>

// =======================================================================================
//                                      CollisionLayer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # CollisionLayer
/// 
/// 11-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

namespace CollisionLayer
{
	enum CollisionLayerType
	{
		COL_NOTHING = 0, //<Collide with nothing
		COL_DEFAULT = BIT(0), //<Collide with default
		COL_GROUND = BIT(1), //<Collide with ground
		COL_CHARACTER = BIT(2) //<Collide with characters
	};

	static const int charactersCollidesWith = COL_GROUND;
	// const int wallCollidesWith = COL_NOTHING;
	// const int powerupCollidesWith = COL_SHIP | COL_WALL;
};