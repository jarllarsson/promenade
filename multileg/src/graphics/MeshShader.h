#pragma once

#include "ShaderVariableContainer.h"
#include "ShaderBase.h"
#include "Buffer.h"

// =======================================================================================
//										MeshShader
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Shader for rendering mesh pixel
///        
/// # MeshShader
/// 
/// 2-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class MeshShader : public ShaderBase
{
public:
	MeshShader(ShaderVariableContainer p_initData);
	virtual ~MeshShader();

	void apply();
protected:
private:
};