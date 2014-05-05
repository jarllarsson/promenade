//#pragma pack_matrix( row_major )
cbuffer EveryFrame : register(b0)
{
	float4x4 gVP;
};

struct VertexIn
{
	// Per vertex
	float3 position : POSITION;
	// Per instance
	float4x4 instanceTransform : INSTANCETRANSFORM;
};

struct VertexOut
{
    float4 position	: SV_POSITION;
};

VertexOut VS(VertexIn p_input)
{
	VertexOut vout;
	float4x4 WVP = mul(p_input.instanceTransform,gVP);
	vout.position = mul(float4(p_input.position, 1),WVP);
    
	return vout;
}

float4 PS(VertexOut input) : SV_TARGET
{
	//float4(1,0.6470588235294118,0,0.5f);
	return float4(0.7333,0.99,0.45f,1.0f);
}