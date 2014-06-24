#pragma once
// #define WIN32_LEAN_AND_MEAN
// #define NOMINMAX
// #include <windows.h>
// 
// #include <d3d11.h>
// 
// #include <directxmath.h>

#include <memory>
#include <PrimitiveBatch.h>
#include <ColorPalettes.h>
#include <glm\gtc\type_ptr.hpp>
#include <vector>
#include <VertexTypes.h>
#include <Effects.h>
#include <wrl\client.h>
#include "DebugDrawBatch.h"
// 
class TempController;

// =======================================================================================
//                                      DebugDrawer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # DebugDrawerSystem
/// 
/// 24-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class DebugDrawer
{
public:
	DebugDrawer(void* p_device, void* p_deviceContext, DebugDrawBatch* p_batch);

	virtual ~DebugDrawer();

	virtual void render(TempController* p_camera);

	void setDrawArea(float p_drawAreaW, float p_drawAreaH);

protected:

private:
	ID3D11DeviceContext* m_deviceContext;
	ID3D11Device* m_device;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_inputLayout;
	float m_drawAreaW, m_drawAreaH;

	DebugDrawBatch* m_batch;

	// Line primitive
	struct XMLine
	{
		DirectX::VertexPositionColor m_start;
		DirectX::VertexPositionColor m_end;
	};
	XMLine getXMLine(const DebugDrawBatch::Line& p_line);

	// Primitive drawing
	std::unique_ptr<DirectX::BasicEffect>                         m_batchEffect;
	std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_primitiveBatch;
};
