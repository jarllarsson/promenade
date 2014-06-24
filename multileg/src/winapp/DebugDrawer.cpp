#include "DebugDrawer.h"
#include <memory>
#include <PrimitiveBatch.h>
#include <VertexTypes.h>
#include <ColorPalettes.h>
#include <glm\gtc\type_ptr.hpp>
#include <vector>
#include <Effects.h>
#include <wrl\client.h>
#include "TempController.h"
#include <Util.h>


DebugDrawer::DebugDrawer(void* p_device, void* p_deviceContext)
{
	m_deviceContext = (ID3D11DeviceContext*)p_deviceContext;
	m_device = (ID3D11Device*)p_device;

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
	DirectX::XMVECTOR v1 = DirectX::XMVectorSet(p_start.x, p_start.y, p_start.z, 0.0f);
	DirectX::XMVECTOR v2 = DirectX::XMVectorSet(p_end.x, p_end.y, p_end.z, 0.0f);
	DirectX::XMVECTOR c1 = DirectX::XMVectorSet(p_startColor.r, p_startColor.g, p_startColor.b, p_startColor.a);
	DirectX::XMVECTOR c2 = DirectX::XMVectorSet(p_endColor.r, p_endColor.g, p_endColor.b, p_endColor.a);
	Line line = { DirectX::VertexPositionColor(v1, c1), DirectX::VertexPositionColor(v2, c2) };
	m_lineList.push_back(line);
}

void DebugDrawer::render(TempController* p_camera)
{
	glm::vec4 camPos = p_camera->getPos();
	glm::quat camRot = glm::inverse(p_camera->getRotation());
	glm::vec3 dir = MathHelp::transformDirection(glm::mat4_cast(camRot), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::vec3 cUp = MathHelp::transformDirection(glm::mat4_cast(camRot), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::vec4 camPosInFront = camPos + glm::vec4(dir.x, dir.y, dir.z, 0.0f);
	glm::vec4 camPosUp = glm::vec4(cUp.x, cUp.y, cUp.z, 0.0f);
	DirectX::XMMATRIX translation = DirectX::XMMatrixTranslation(camPos.x, camPos.y, camPos.z);
	DirectX::XMMATRIX rotation = DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(camRot.x, camRot.y, camRot.z, camRot.w));
	DirectX::XMMATRIX camTransform = DirectX::XMMatrixMultiply(

		DirectX::XMMatrixInverse(&XMMatrixDeterminant(translation), translation), rotation/*




		*/);
	/*camTransform = DirectX::XMMatrixTransformation(DirectX::g_XMZero, DirectX::XMQuaternionIdentity(), 
		DirectX::g_XMOne,
		DirectX::XMQuaternionIdentity(),
		DirectX::XMVectorSet(camRot.x, camRot.y, camRot.z, camRot.w),
		DirectX::XMVectorSet(-camPos.x, -camPos.y, camPos.z, 1.0f));*/

	camTransform = DirectX::XMMatrixLookAtLH(DirectX::XMVectorSet(camPos.x, camPos.y, camPos.z, 1.0f),
		DirectX::XMVectorSet(camPosInFront.x, camPosInFront.y, camPosInFront.z, 1.0f),
		DirectX::XMVectorSet(camPosUp.x, camPosUp.y, camPosUp.z, 0.0f));

	//m_batchEffect->SetView(DirectX::XMMatrixInverse(&XMMatrixDeterminant(camTransform),camTransform));	
	DirectX::XMMATRIX camProj = DirectX::XMMatrixPerspectiveFovLH(p_camera->getFovAngle()*(float)TORAD,
		m_drawAreaW / m_drawAreaH, 0.1f, 1000.0f);
	m_batchEffect->SetProjection(camProj);

	//m_batchEffect->SetWorld();
	m_batchEffect->SetView(camTransform);



	m_batchEffect->Apply(m_deviceContext);
	m_deviceContext->IASetInputLayout(m_inputLayout.Get());

	// Draw debug primitives
	m_primitiveBatch->Begin();
	// Draw all queued lines:
	for (int i = 0; i < m_lineList.size(); i++)
	{
		Line* line = &m_lineList[i];
		m_primitiveBatch->DrawLine(line->m_start, line->m_end);
	}
	m_primitiveBatch->End();
	// End of drawing debug primitives

	clearLineList();
}

void DebugDrawer::clearLineList()
{
	m_lineList.clear();
}

void DebugDrawer::setDrawArea(float p_drawAreaW, float p_drawAreaH)
{
	m_drawAreaW = p_drawAreaW; m_drawAreaH = p_drawAreaH;
}
