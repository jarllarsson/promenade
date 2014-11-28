// -------------------------------------------
// Buffer with data used on a per-frame basis
// -------------------------------------------
cbuffer EveryFrame : register(b0)
{
	float4x4 gVP;
};