#include "MeshSceneDefines.hlsl"

struct PixelOut
{
	float4 diffuse	: SV_TARGET0;		//diffuse
	float4 normal	: SV_TARGET1;		//normal
	//float4 specular : SV_TARGET2;		//specular
};

struct VertexIn
{
	// Per vertex
	float3 position : POSITION;
	float3 normal : NORMAL;
	// Per instance
	float4x4 instanceTransform : INSTANCETRANSFORM;
	float4 instanceColor	: INSTANCECOLOR;
};

struct VertexOut
{
    float4 position	: SV_POSITION;
	float3 normal : NORMAL;
	float4 color	: COLOR;
};

VertexOut VS(VertexIn p_input)
{
	VertexOut vout;
	float4x4 W = p_input.instanceTransform;
	float4x4 WVP = mul(W,gVP);
	vout.position = mul(float4(p_input.position, 1),WVP);
	vout.normal = normalize(mul(float4(p_input.normal, 0), W).xyz);
    vout.color = p_input.instanceColor;
	return vout;
}

PixelOut PS(VertexOut p_input)
{
	PixelOut pixelOut;
	//float4(1,0.6470588235294118,0,0.5f);
	//return float4(0.7333,0.99,0.45f,1.0f);
	pixelOut.diffuse = p_input.color;
	pixelOut.normal = float4(p_input.normal,0);
	//pixelOut.specular = float4(0,1,0,1);
	return pixelOut;
}






// ============================================

/*
// Total of 288 bytes
cbuffer perFrame: register(b0)
{
	float4x4	gView;						//64 bytes
	float4x4 	gViewProj;					//64 bytes 	
	// float4x4 	gViewProjInverse;			//64 bytes	
	// float4		gCameraPos;					//16 bytes	
	// float4		gCameraForward;				//16 bytes	
	// float4		gCameraUp;					//16 bytes	
	// float4		gAmbientColorAndFogNear;	//16 bytes	
	// float4		gFogColorAndFogFar;			//16 bytes
	// float2		gRenderTargetSize;			//8 bytes	
	// float		gFarPlane;					//4 bytes	
	// float		gNearPlane;					//4 bytes	
};

SamplerState g_sampler : register(s0);

struct VertexIn
{
	// Per vertex
	float3 position : POSITION;
	// float3 normal 	: NORMAL;
	// float2 texCoord : TEXCOORD; 
	// float3 tangent 	: TANGENT;	
	// float3 binormal : BINORMAL;
	
	//Per instance
	float4x4 instanceTransform  : INSTANCETRANSFORM;
	// float4x4 gradientColor 		: GRADIENTCOLOR;
	// float4 flags 	: FLAGS;
	// float4 colorOverlay : OVERLAY;
};
struct VertexOut
{
    float4 position	: SV_POSITION;
	float3 color : COLOR;
};

VertexOut VS(VertexIn p_input)
{
	VertexOut vout;
	float4x4 wvp = mul(p_input.instanceTransform,gViewProj);
	
	vout.position = mul(float4(p_input.position,1.0f), wvp);
	// vout.normal = mul(float4(p_input.normal,0.0f), p_input.instanceTransform).xyz;
	// vout.tangent = mul(float4(p_input.tangent,0.0f),p_input.instanceTransform).xyz;
	// vout.texCoord = p_input.texCoord;
	// vout.gradientColor = p_input.gradientColor;
	// vout.flags = p_input.flags;
	// vout.colorOverlay = p_input.colorOverlay;
	
	// vout.position = float4(p_input.position,1.0f);
	vout.color = (vout.position+1.0f)*0.5f;
    
	return vout;
}

float4 PS(VertexOut p_input) : SV_TARGET
{

	float3 finalCol = p_input.color;
	//finalCol.r *= input.color.r;
	//finalCol.b *= input.color.g;
	
	// test
	// finalCol.rgb = float3(0.0f,0.0f,1.0f);
	
	return float4( finalCol.rgb, 1.0f );
}
*/