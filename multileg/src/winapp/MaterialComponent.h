#pragma once
#include <Artemis.h>
#include <ColorPalettes.h>
// =======================================================================================
//                                      MaterialComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Component describing the material to render on a renderable
///        
/// # MaterialComponent
/// 
/// 18-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class MaterialComponent : public artemis::Component
{
public:
	MaterialComponent();
	MaterialComponent(const Color4f& p_color);
	MaterialComponent(const Color3f& p_color);
	~MaterialComponent() {}

	void setColorRGB(const Color3f& p_color);
	void setColorRGBA(const Color4f& p_color);

	const Color4f& getColorRGBA();
	const Color3f getColorRGB();

	bool isMaterialRenderDirty();
	void unsetMaterialRenderDirty();

	void highLight();
	void unsetHighLight();

protected:
private:
	void init();
	bool m_dirty;
	bool m_highLightFac;
	Color4f m_color;
};