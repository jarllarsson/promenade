// -------------------------------------------
// Buffer with data used on a per-frame basis
// -------------------------------------------
cbuffer EveryFrame : register(b0)
{
	float4x4 gVP;
};

// -------------------------------------------
// Buffer with data used on a per-object basis
// -------------------------------------------
cbuffer PerObject
{
	float4x4 gWorldMat;	// A World matrix
	float4x4 gWorldViewMat;	// A World-view matrix
	float4x4 gWVP; 		// A World-View-Projection matrix
	float4x4 gPrevWVP; 		// the old World-View-Projection matrix
	float4x4 gTextureMat;	// Texture transformation matrix
};

// -------------------------------------------
// Buffer with data used on a per-material basis
// -------------------------------------------
cbuffer PerMaterial
{
	float4 gDiffuseColor; // base diffuse color
	float4 gSpecularColor; // base specular color
};