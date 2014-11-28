#pragma once
#include <windows.h>
#include <d3d11.h>
#include "Buffer.h"
#include "RenderStateHelper.h"
#include <RunLengthList.h>

struct PVertex;
class ViewFactory;
class BufferFactory;
class ShaderFactory;
class TextureFactory;
class ComposeShader;
class MeshShader;
class Texture;
class Mesh;



// =======================================================================================
//                                     GraphicsDevice
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class for all things graphics, allocation and rendering
///        
/// # GraphicsDevice
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class GraphicsDevice
{
public:
	const static int RT0 = 0;
	const static int RT1 = 1;
	const static int DEPTH_IDX = 10;
	enum GBufferChannel {
		GBUF_INVALID	= -1,
		GBUF_DIFFUSE	= RT0,				// R, G, B, LinDepth(Raytracer)
		GBUF_NORMAL		= RT1,				// X, Y, Z		
		GBUF_COUNT,
		GBUF_DEPTH		= DEPTH_IDX,		// Depth
	};

	enum RenderTargetSpec {
		RT_NONE,
		RT_BACKBUFFER,
		RT_MRT,
		RT_BACKBUFFER_NODEPTHSTENCIL,
		RT_MRT_NODEPTHSTENCIL,
		RT_COUNT,
	};

	enum ShaderId {
		SI_NONE,
		SI_MESHBASESHADER,
		SI_COMPOSESHADER,
		SI_WIREFRAMESHADER,
		SI_COUNT,
	};

	enum RenderPass {
		P_BASEPASS,
		P_COMPOSEPASS,
		P_BOUNDINGBOX_WIREFRAMEPASS,
		P_COUNT,
	};

	GraphicsDevice(HWND p_hWnd, int p_width, int p_height, bool p_windowMode);
	virtual ~GraphicsDevice();

	// States
	// Clear render targets in a color
	void clearRenderTargets();								///< Clear all rendertargets
	void flipBackBuffer();									///< Fliparoo!

	void updateResolution( int p_width, int p_height );		///< Update resolution
	void setWindowMode(bool p_windowed);					///< Set window mode on/off
	void fitViewport();										///< Fit viewport to width and height
	void setWireframeMode( bool p_wireframe );				///< Force wireframe render

	// Info
	float getAspectRatio();
	int getWidth();
	int getHeight();

	// Stages
	void executeRenderPass(RenderPass p_pass,
						   BufferBase* p_cbuf=NULL, 
						   vector<BufferBase*>* p_instancesLists=NULL,
						   vector<Mesh*>* p_meshList=NULL);


	// Getters
	void* getDevicePointer();
	void* getDeviceContextPointer();
	void** getInteropCanvasHandle();
	BufferFactory* getBufferFactoryRef();

protected:
private:	
	// Mapping/Unmapping
	void mapGBuffer();
	void mapGBufferSlot(GBufferChannel p_slot);
	void mapDepth();

	void unmapGBuffer();
	void unmapGBufferSlot(GBufferChannel p_slot);
	void unmapDepth();
	void unmapAllBuffers();

	// Rendertarget set
	void setRenderTarget(RenderTargetSpec p_target);

	// Shader set
	void setShader(ShaderId p_shaderId);

	// Blend states
	void setBlendState(BlendState::Mode p_state);
	void setBlendFactors(float p_red, float p_green, float p_blue, float p_alpha);
	void setBlendFactors(float p_oneValue);
	void setBlendMask(UINT p_mask);
	BlendState::Mode getCurrentBlendStateType();


	// Rasterizer states
	void setRasterizerStateSettings(RasterizerState::Mode p_state,
									bool p_allowWireframOverride=true);
	RasterizerState::Mode getCurrentRasterizerStateType();

	// Draw
	void drawFullscreen();
	void drawInstancedLineOBB(UINT32 p_instanceElementCount, int p_instanceElemSz, void* p_instanceRef);
	void drawInstancedIndexedMesh(Mesh* p_mesh, UINT32 p_instanceElementCount, int p_instanceElemSz, void* p_instanceRef);

	// Initialisations
	void initSwapChain(HWND p_hWnd);
	void initHardware();	
	void initBackBuffer();
	void initDepthStencil();
	void initGBuffer();
	void initGBufferAndDepthStencil();
	void buildBlendStates();
	void buildRasterizerStates();
	// Releases
	void releaseBackBuffer();
	void releaseGBufferAndDepthStencil();

	// Members
	int m_height;
	int m_width;
	bool m_windowMode;
	bool m_wireframeMode;

	// Factories
	ViewFactory* m_viewFactory;
	ShaderFactory* m_shaderFactory;
	BufferFactory* m_bufferFactory;
	TextureFactory* m_textureFactory;

	// Shaders
	ComposeShader* m_composeShader;
	MeshShader* m_wireframeShader;

	// Fullscreen quad for drawing
	Buffer<PVertex>* m_fullscreenQuad;
	Buffer<PVertex>* m_aabbLineMesh;

	// Blend states
	vector<ID3D11BlendState*> m_blendStates;
	BlendState::Mode m_currentBlendStateType;
	float m_blendFactors[4];
	UINT m_blendMask;

	// Rasterizer states
	vector<ID3D11RasterizerState*> m_rasterizerStates;
	RasterizerState::Mode m_currentRasterizerStateType;

	// D3D specific
	// device
	ID3D11Device*			m_device;
	ID3D11DeviceContext*	m_deviceContext;
	// swap chain
	DXGI_SWAP_CHAIN_DESC	m_swapChainDesc;
	IDXGISwapChain*			m_swapChain;
	D3D_FEATURE_LEVEL		m_featureLevel;
	// views
	ID3D11RenderTargetView*		m_backBuffer;
	ID3D11ShaderResourceView*	m_depthSrv;
	ID3D11DepthStencilView*		m_depthStencilView;
	Texture*					m_gTexture[GBufferChannel::GBUF_COUNT];
	Texture**					m_interopCanvasHandle; ///< Handle enabling texture reallocation
	ID3D11RenderTargetView*		m_gRtv[GBufferChannel::GBUF_COUNT];
	ID3D11ShaderResourceView*	m_gSrv[GBufferChannel::GBUF_COUNT];

};