#pragma once

// =======================================================================================
//                                      MathHelp
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # MathHelp
/// 
/// 24-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------
#define PI 3.141592653589793238462643383279502884197169399375105820
#define TWOPI 2.0*PI
#define TORAD PI/180
#define TODEG 180/PI
#define PIOVER180 TORAD


static size_t roundup(int group_size, int global_size) 
{
	int r = global_size % group_size;
	if(r == 0) 
	{
		return global_size;
	} else 
	{
		return global_size + group_size - r;
	}
}
