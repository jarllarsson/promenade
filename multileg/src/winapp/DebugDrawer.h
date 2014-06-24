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
	DebugDrawer(void* p_device, void* p_deviceContext);

	virtual ~DebugDrawer();

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_color);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_color);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_startColor, const Color3f& p_endColor);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_startColor, const Color4f& p_endColor);

	virtual void render(TempController* p_camera);

	void setDrawArea(float p_drawAreaW, float p_drawAreaH);

protected:

private:
	ID3D11DeviceContext* m_deviceContext;
	ID3D11Device* m_device;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_inputLayout;
	float m_drawAreaW, m_drawAreaH;

	// Line primitive
	struct Line
	{
		DirectX::VertexPositionColor m_start;
		DirectX::VertexPositionColor m_end;
	};
	std::vector<Line> m_lineList;
	void clearLineList();

	// Primitive drawing
	std::unique_ptr<DirectX::BasicEffect>                         m_batchEffect;
	std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_primitiveBatch;
};
