//#pragma pack_matrix( column_major )
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
	float4x4 WVP = mul(gVP,p_input.instanceTransform);
	vout.position = mul(WVP,float4(p_input.position, 1));
    
	return vout;
}

float4 PS(VertexOut input) : SV_TARGET
{
	return float4(1,0.6470588235294118,0,1);
}