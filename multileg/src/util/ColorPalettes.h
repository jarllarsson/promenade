#pragma once

struct Color3f
{
	Color3f()
	{
		r = 0.0f; g = 0.0f; b = 0.0f;
	}
	Color3f(float p_r, float p_g, float p_b)
	{
		r = p_r; g = p_g; b = p_b;
	}
	float r, g, b;
	Color3f operator* (float x) const
	{
		Color3f ncol{ r * x, g * x, b * x};
		return ncol;
	}
};

struct Color4f
{
	Color4f()
	{
		r = 0.0f; g = 0.0f; b = 0.0f; a = 1.0f;
	}
	Color4f(float p_r, float p_g, float p_b, float p_a=1.0f)
	{
		r = p_r; g = p_g; b = p_b; a = p_a;
	}
	float r, g, b, a;
	Color4f operator* (float x) const
	{
		Color4f ncol{ r * x, g * x, b * x, a * x };		
		return ncol;
	}
};

Color4f toColor4f(const Color3f& p_col);

Color3f toColor3f(const Color4f& p_col);

#define colarrSz 18

static const Color3f colarr[colarrSz] = { { 1.0f, 0.0f, 0.0f },
{ 1.0f, 0.5f, 0.0f },
{ 1.0f, 0.0f, 0.5f },
{ 0.0f, 0.7f, 0.0f },
{ 0.0f, 0.7f, 0.3f },
{ 0.3f, 0.7f, 0.0f },
{ 0.2f, 1.0f, 0.42f },
{ 0.5f, 1.0f, 0.0f },
{ 1.0f, 0.0f, 0.5f },
{ 0.5f, 0.5f, 0.5f },
{ 1.0f, 0.7f, 0.7f },
{ 1.0f, 1.0f, 0.2f },
{ 0.33f, 0.25f, 0.4f },
{ 0.8f, 1.0f, 0.24f },
{ 0.0f, 0.66f, 0.5f },
{ 0.1f, 0.231f, 0.13f },
{ 0.87f, 0.5f, 1.0f },
{ 0.2f, 0.2f, 1.0f } };

// indices for the dawnbringer palette
#define COL_DARKGREY 0
#define COL_DARKPURPLE 1
#define COL_DARKBROWN 2
#define COL_DARKBEIGE 3
#define COL_ORANGE 4
#define COL_BEIGE 5
#define COL_LIGHTBEIGE 6
#define COL_YELLOW 7
#define COL_SLIMEGREEN 8
#define COL_LIMEGREEN 9
#define COL_TURQUOISE 10
#define COL_DARKGREEN 11
#define COL_DARKFOREST 12
#define COL_DARKOCEAN 13
#define COL_NAVALBLUE 14
#define COL_DARKBLUE 15
#define COL_CORNFLOWERBLUE 16
#define COL_LIGHTCORNFLOWERBLUE 17
#define COL_LIGHTBLUE 18
#define COL_LIGHTBLUEGREY 19
#define COL_WHITE 20
#define COL_GREY 21
#define COL_CONCRETE 22
#define COL_DARKCONCRETE 23
#define COL_ASPHALT 24
#define COL_PURPLE 25
#define COL_RED 26
#define COL_LIGHTRED 27
#define COL_PINK 28
#define COL_MOSSGREEN 29
#define COL_BARF 30

// Dawnbringer 32-col palette (I removed pure black)
// http://www.pixeljoint.com/forum/forum_posts.asp?TID=16247
static const Color3f dawnBringerPalRGB[31] = {
		{ 34 / 256.0f, 32 / 256.0f, 52 / 256.0f },
		{ 69 / 256.0f, 40 / 256.0f, 60 / 256.0f },
		{ 102 / 256.0f, 57 / 256.0f, 49 / 256.0f },
		{ 143 / 256.0f, 86 / 256.0f, 59 / 256.0f },
		{ 223 / 256.0f, 113 / 256.0f, 38 / 256.0f },
		{ 217 / 256.0f, 160 / 256.0f, 102 / 256.0f },
		{ 238 / 256.0f, 195 / 256.0f, 154 / 256.0f },
		{ 251 / 256.0f, 242 / 256.0f, 54 / 256.0f },
		{ 153 / 256.0f, 229 / 256.0f, 80 / 256.0f },
		{ 106 / 256.0f, 190 / 256.0f, 48 / 256.0f },
		{ 55 / 256.0f, 148 / 256.0f, 110 / 256.0f },
		{ 75 / 256.0f, 105 / 256.0f, 47 / 256.0f },
		{ 82 / 256.0f, 75 / 256.0f, 36 / 256.0f },
		{ 50 / 256.0f, 60 / 256.0f, 57 / 256.0f },
		{ 63 / 256.0f, 63 / 256.0f, 116 / 256.0f },
		{ 48 / 256.0f, 96 / 256.0f, 130 / 256.0f },
		{ 91 / 256.0f, 110 / 256.0f, 225 / 256.0f },
		{ 99 / 256.0f, 155 / 256.0f, 255 / 256.0f },
		{ 95 / 256.0f, 205 / 256.0f, 228 / 256.0f },
		{ 203 / 256.0f, 219 / 256.0f, 252 / 256.0f },
		{ 255 / 256.0f, 255 / 256.0f, 255 / 256.0f },
		{ 155 / 256.0f, 173 / 256.0f, 183 / 256.0f },
		{ 132 / 256.0f, 126 / 256.0f, 135 / 256.0f },
		{ 105 / 256.0f, 106 / 256.0f, 106 / 256.0f },
		{ 89 / 256.0f, 86 / 256.0f, 82 / 256.0f },
		{ 118 / 256.0f, 66 / 256.0f, 138 / 256.0f },
		{ 172 / 256.0f, 50 / 256.0f, 50 / 256.0f },
		{ 217 / 256.0f, 87 / 256.0f, 99 / 256.0f },
		{ 215 / 256.0f, 123 / 256.0f, 186 / 256.0f },
		{ 143 / 256.0f, 151 / 256.0f, 74 / 256.0f },
		{ 138 / 256.0f, 111 / 256.0f, 48 / 256.0f }
};

// Same, but with alpha channel as well. CONVENIENT ;D
static const Color4f dawnBringerPalRGBA[31] = {
		{ 34 / 256.0f, 32 / 256.0f, 52 / 256.0f		,1.0f},
		{ 69 / 256.0f, 40 / 256.0f, 60 / 256.0f		,1.0f},
		{ 102 / 256.0f, 57 / 256.0f, 49 / 256.0f	,1.0f},
		{ 143 / 256.0f, 86 / 256.0f, 59 / 256.0f	,1.0f},
		{ 223 / 256.0f, 113 / 256.0f, 38 / 256.0f	,1.0f},
		{ 217 / 256.0f, 160 / 256.0f, 102 / 256.0f	,1.0f},
		{ 238 / 256.0f, 195 / 256.0f, 154 / 256.0f	,1.0f},
		{ 251 / 256.0f, 242 / 256.0f, 54 / 256.0f	,1.0f},
		{ 153 / 256.0f, 229 / 256.0f, 80 / 256.0f	,1.0f},
		{ 106 / 256.0f, 190 / 256.0f, 48 / 256.0f	,1.0f},
		{ 55 / 256.0f, 148 / 256.0f, 110 / 256.0f	,1.0f},
		{ 75 / 256.0f, 105 / 256.0f, 47 / 256.0f	,1.0f},
		{ 82 / 256.0f, 75 / 256.0f, 36 / 256.0f		,1.0f},
		{ 50 / 256.0f, 60 / 256.0f, 57 / 256.0f		,1.0f},
		{ 63 / 256.0f, 63 / 256.0f, 116 / 256.0f	,1.0f},
		{ 48 / 256.0f, 96 / 256.0f, 130 / 256.0f	,1.0f},
		{ 91 / 256.0f, 110 / 256.0f, 225 / 256.0f	,1.0f},
		{ 99 / 256.0f, 155 / 256.0f, 255 / 256.0f	,1.0f},
		{ 95 / 256.0f, 205 / 256.0f, 228 / 256.0f	,1.0f},
		{ 203 / 256.0f, 219 / 256.0f, 252 / 256.0f	,1.0f},
		{ 255 / 256.0f, 255 / 256.0f, 255 / 256.0f	,1.0f},
		{ 155 / 256.0f, 173 / 256.0f, 183 / 256.0f	,1.0f},
		{ 132 / 256.0f, 126 / 256.0f, 135 / 256.0f	,1.0f},
		{ 105 / 256.0f, 106 / 256.0f, 106 / 256.0f	,1.0f},
		{ 89 / 256.0f, 86 / 256.0f, 82 / 256.0f		,1.0f},
		{ 118 / 256.0f, 66 / 256.0f, 138 / 256.0f	,1.0f},
		{ 172 / 256.0f, 50 / 256.0f, 50 / 256.0f	,1.0f},
		{ 217 / 256.0f, 87 / 256.0f, 99 / 256.0f	,1.0f},
		{ 215 / 256.0f, 123 / 256.0f, 186 / 256.0f	,1.0f},
		{ 143 / 256.0f, 151 / 256.0f, 74 / 256.0f	,1.0f},
		{ 138 / 256.0f, 111 / 256.0f, 48 / 256.0f	,1.0f}
};