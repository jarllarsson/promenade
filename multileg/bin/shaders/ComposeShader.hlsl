Texture2D g_diffuse				: register(t0);
Texture2D g_normal				: register(t1);

Texture2D g_depth				: register(t10);

SamplerState g_samplerPointWrap : register(s0);

struct VertexIn
{
	float3 position : POSITION;
};
struct VertexOut
{
    float4 position	: SV_POSITION;
	//float3 color : COLOR;
};

VertexOut VS(VertexIn p_input)
{
	VertexOut vout;
	vout.position = float4(p_input.position,1.0f);
	//vout.color = (p_input.position+1.0f)*0.5f;
    
	return vout;
}

float4 PS(VertexOut input) : SV_TARGET
{
	uint3 index;
	index.xy = input.position.xy;
	index.z = 0;
	float3 normal = g_normal.Load(index).xyz;
	float4 diffuse = g_diffuse.Load(index);
	
	// the light
	float3 lightdir = -normalize(float3(0.5f,-1.0f,0.5f));
	float3 lightcol = float3(1,1,1);
	
	// wrap lambert
	float3 NdotL = dot(normal, lightdir);
	float3 diff = NdotL * 0.5f + 0.5f;
	float atten = 1.0f;
	float3 c = diffuse.rgb;
	if (diffuse.a>=0.0f)
		c *= lightcol * (diff*atten*2.0f);
	
	float3 finalCol = c;
	//finalCol.r *= input.color.r;
	//finalCol.b *= input.color.g;
	
	return float4( finalCol.rgb, 1.0f );
}