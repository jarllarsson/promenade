#pragma once

#include "ShaderVariableContainer.h"
#include "ShaderBase.h"
#include "Buffer.h"

// =======================================================================================
//                                      ComposeShader
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Shader for final composition of MRT
///        
/// # ComposeShader
/// 
/// 19-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class ComposeShader : public ShaderBase
{
public:
	ComposeShader(ShaderVariableContainer p_initData);
	virtual ~ComposeShader();

	void apply();
protected:
private:
};