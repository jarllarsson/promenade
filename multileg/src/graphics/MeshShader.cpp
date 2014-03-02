#include "MeshShader.h"


MeshShader::MeshShader( ShaderVariableContainer p_initData )
	: ShaderBase(p_initData)
{

}

MeshShader::~MeshShader()
{

}

void MeshShader::apply()
{
	applyStages();
}
