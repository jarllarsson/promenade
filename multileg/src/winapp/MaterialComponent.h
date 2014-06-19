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
	MaterialComponent() { setColorRGB({ 1, 1, 1 }); }
	MaterialComponent(const Color4f& p_color) { setColorRGBA(p_color); }
	MaterialComponent(const Color3f& p_color) { setColorRGB(p_color); }
	~MaterialComponent() {}

	void setColorRGB(const Color3f& p_color)
	{
		setColorRGBA(toColor4f(p_color));
	}
	void setColorRGBA(const Color4f& p_color)
	{
		m_color = p_color;
		m_dirty = true;
	}
	const Color4f& getColorRGBA() { return m_color; }
	const Color3f& getColorRGB() { return toColor3f(m_color); }

	bool isMaterialRenderDirty() { return m_dirty; }
	void unsetMaterialRenderDirty() { m_dirty = false; }

protected:
private:
	bool m_dirty;
	Color4f m_color;
};