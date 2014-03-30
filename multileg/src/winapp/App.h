#pragma once

#include <windows.h>
#include <glm\gtc\type_ptr.hpp>

#include <InstanceData.h>
#include <CBuffers.h>
#include <Buffer.h>
#include <vector>

class Context;
class GraphicsDevice;
class KernelDevice;
class TempController;
class OISHelper;

using namespace std;

// =======================================================================================
//                                      App
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # App
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class App
{
public:
	App(HINSTANCE p_hInstance);
	virtual ~App();

	void run();
protected:
	Context* m_context;
	GraphicsDevice* m_graphicsDevice;
	KernelDevice* m_kernelDevice;
private:
	static const double DTCAP;
	float fpsUpdateTick;

	void updateController();

	TempController* m_controller;
	OISHelper*		m_input;

	vector<glm::mat4> m_instance;
	Buffer<InstanceData>* m_instances;
	Buffer<Mat4CBuffer>* m_vp;
};