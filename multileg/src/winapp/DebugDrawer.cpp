#include "DebugDrawer.h"
#include <GraphicsDevice.h>
#include <memory>
#include <PrimitiveBatch.h>
#include <VertexTypes.h>
#include <ColorPalettes.h>
#include <glm\gtc\type_ptr.hpp>
#include <vector>
#include <Effects.h>
#include <wrl\client.h>
#include "TempController.h"



DebugDrawer::DebugDrawer(GraphicsDevice* p_graphicsDevice)
{
	m_graphicsDevice = p_graphicsDevice;
	m_deviceContext = (ID3D11DeviceContext*)m_graphicsDevice->getDeviceContextPointer();
	m_device = (ID3D11Device*)m_graphicsDevice->getDevicePointer();

	m_primitiveBatch = std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>>(
		new DirectX::PrimitiveBatch<DirectX::VertexPositionColor>(m_deviceContext));
	m_batchEffect = std::unique_ptr<DirectX::BasicEffect>(new DirectX::BasicEffect(m_device));

	m_batchEffect->SetVertexColorEnabled(true);

	void const* shaderByteCode;
	size_t byteCodeLength;

	m_batchEffect->GetVertexShaderBytecode(&shaderByteCode, &byteCodeLength);

	m_device->CreateInputLayout(DirectX::VertexPositionColor::InputElements,
		DirectX::VertexPositionColor::InputElementCount,
		shaderByteCode, byteCodeLength,
		m_inputLayout.GetAddressOf());
}

DebugDrawer::~DebugDrawer()
{

}

void DebugDrawer::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color4f& p_color)
{
	drawLine(p_start, p_end, p_color, p_color);
}
void DebugDrawer::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color3f& p_color)
{
	drawLine(p_start, p_end, toColor4f(p_color), toColor4f(p_color));
}
void DebugDrawer::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color3f& p_startColor, const Color3f& p_endColor)
{
	drawLine(p_start, p_end, toColor4f(p_startColor), toColor4f(p_endColor));
}
void DebugDrawer::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color4f& p_startColor, const Color4f& p_endColor)
{
	//DirectX::FXMVECTOR v1 = DirectX::XMVectorSet(p_start.x, p_start.y, p_start.z, 0.0f);
	//DirectX::FXMVECTOR v2 = DirectX::XMVectorSet(p_end.x, p_end.y, p_end.z, 0.0f);
	//DirectX::FXMVECTOR c1 = DirectX::XMVectorSet(p_startColor.r, p_startColor.g, p_startColor.b, p_startColor.a);
	//DirectX::FXMVECTOR c2 = DirectX::XMVectorSet(p_endColor.r, p_endColor.g, p_endColor.b, p_endColor.a);
	//Line line = { DirectX::VertexPositionColor(v1, c1), DirectX::VertexPositionColor(v2, c2) };
	//m_lineList.push_back(line);
}

void DebugDrawer::render(TempController* p_camera)
{
	glm::vec4 camPos = p_camera->getPos();
	glm::quat camRot = p_camera->getRotation();
	//DirectX::FXMMATRIX camTransform = DirectX::XMMatrixMultiply(
	//	DirectX::XMMatrixTranslation(camPos.x, camPos.y, camPos.z),
	//	DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(camRot.x, camRot.y, camRot.z, camRot.w)));
	//m_batchEffect->SetView(camTransform);
	//DirectX::FXMMATRIX camProj = DirectX::XMMatrixPerspectiveFovRH(p_camera->getFovXY().y,
	//	p_camera->getAspect(), 0.001f, 1000.0f);
	//m_batchEffect->SetProjection(camProj);

	m_batchEffect->Apply(m_deviceContext);
	m_deviceContext->IASetInputLayout(m_inputLayout.Get());

	// Draw debug primitives
	m_primitiveBatch->Begin();
	// Draw all queued lines:
	for (int i = 0; i < m_lineList.size(); i++)
	{
		//Line* line = &m_lineList[i];
		//m_primitiveBatch->DrawLine(line->m_start, line->m_end);
	}
	m_primitiveBatch->End();
	// End of drawing debug primitives

	m_lineList.clear();
}
