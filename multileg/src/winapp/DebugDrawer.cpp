#include "DebugDrawer.h"
// #include <memory>
// #include <PrimitiveBatch.h>
// #include <VertexTypes.h>
// #include <ColorPalettes.h>
// #include <glm\gtc\type_ptr.hpp>
// #include <vector>
// #include <Effects.h>
// #include <wrl\client.h>
#include "TempController.h"
#include <Util.h>


DebugDrawer::DebugDrawer(void* p_device, void* p_deviceContext, DebugDrawBatch* p_batch)
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

	m_batch = p_batch;
}

DebugDrawer::~DebugDrawer()
{

}


void DebugDrawer::render(TempController* p_camera)
{
	glm::vec4 camPos = p_camera->getPos();
	glm::quat camRot = glm::inverse(p_camera->getRotation());
	glm::vec3 dir = MathHelp::transformDirection(glm::mat4_cast(camRot), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::vec3 cUp = MathHelp::transformDirection(glm::mat4_cast(camRot), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::vec4 camPosInFront = camPos + glm::vec4(dir.x, dir.y, dir.z, 0.0f);
	glm::vec4 camPosUp = glm::vec4(cUp.x, cUp.y, cUp.z, 0.0f);

	DirectX::XMMATRIX camTransform = DirectX::XMMatrixLookAtLH(DirectX::XMVectorSet(camPos.x, camPos.y, camPos.z, 1.0f),
		DirectX::XMVectorSet(camPosInFront.x, camPosInFront.y, camPosInFront.z, 1.0f),
		DirectX::XMVectorSet(camPosUp.x, camPosUp.y, camPosUp.z, 0.0f));

	DirectX::XMMATRIX camProj = DirectX::XMMatrixPerspectiveFovLH(p_camera->getFovAngle()*(float)TORAD,
		m_drawAreaW / m_drawAreaH, 0.1f, 1000.0f);
	m_batchEffect->SetProjection(camProj);

	m_batchEffect->SetView(camTransform);



	m_batchEffect->Apply(m_deviceContext);
	m_deviceContext->IASetInputLayout(m_inputLayout.Get());

	// Draw debug primitives
	m_primitiveBatch->Begin();
	// Draw all queued lines:
	std::vector<DebugDrawBatch::Line>* linelist = m_batch->getLineList();
	for (int i = 0; i < linelist->size(); i++)
	{
		XMLine line = getXMLine((*linelist)[i]);
		m_primitiveBatch->DrawLine(line.m_start, line.m_end);
	}
	// Draw all queued spheres:
	/* TODO
	std::vector<DebugDrawBatch::Line>* linelist = m_batch->getLineList();
	for (int i = 0; i < linelist->size(); i++)
	{
		XMLine line = getXMLine((*linelist)[i]);
		m_primitiveBatch->
	}
	*/
	// End of drawing debug primitives
	m_primitiveBatch->End();
}



void DebugDrawer::setDrawArea(float p_drawAreaW, float p_drawAreaH)
{
	m_drawAreaW = p_drawAreaW; m_drawAreaH = p_drawAreaH;
}

DebugDrawer::XMLine DebugDrawer::getXMLine(const DebugDrawBatch::Line& p_line)
{
	glm::vec3 start = p_line.m_start, end = p_line.m_end;
	Color4f startColor = p_line.m_startColor, endColor = p_line.m_endColor;
	DirectX::XMVECTOR v1 = DirectX::XMVectorSet(start.x, start.y, start.z, 0.0f);
	DirectX::XMVECTOR v2 = DirectX::XMVectorSet(end.x, end.y, end.z, 0.0f);
	DirectX::XMVECTOR c1 = DirectX::XMVectorSet(startColor.r, startColor.g, startColor.b, startColor.a);
	DirectX::XMVECTOR c2 = DirectX::XMVectorSet(endColor.r, endColor.g, endColor.b, endColor.a);
	XMLine line = { DirectX::VertexPositionColor(v1, c1), DirectX::VertexPositionColor(v2, c2) };
	return line;
}
