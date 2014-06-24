#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <d3d11.h>

#include <directxmath.h>

#include <memory>
#include <PrimitiveBatch.h>
#include <VertexTypes.h>
#include <ColorPalettes.h>
#include <glm\gtc\type_ptr.hpp>
#include <vector>
#include <Effects.h>
#include <wrl\client.h>

class GraphicsDevice;
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
	DebugDrawer(GraphicsDevice* p_graphicsDevice);

	virtual ~DebugDrawer();

	void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_color);

	void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_color);

	void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_startColor, const Color3f& p_endColor);

	void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_startColor, const Color4f& p_endColor);

	void render(TempController* p_camera);

protected:
private:
	GraphicsDevice* m_graphicsDevice;
	ID3D11DeviceContext* m_deviceContext;
	ID3D11Device* m_device;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_inputLayout;

	// Line primitive
	struct Line
	{
		//DirectX::VertexPositionColor m_start, m_end;
	};
	std::vector<Line> m_lineList;

	// Primitive drawing
	std::unique_ptr<DirectX::BasicEffect>                         m_batchEffect;
	std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_primitiveBatch;
};