#include "ColorPalettes.h"

Color4f toColor4f(const Color3f& p_col)
{
	return{ p_col.r, p_col.g, p_col.b, 1.0f };
}

Color3f toColor3f(const Color4f& p_col)
{
	return{ p_col.r, p_col.g, p_col.b };
}