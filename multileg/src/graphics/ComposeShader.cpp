#include "ComposeShader.h"

ComposeShader::ComposeShader( ShaderVariableContainer p_initData ) : ShaderBase(p_initData)
{

}

ComposeShader::~ComposeShader()
{

}

void ComposeShader::apply()
{
	applyStages();
}
