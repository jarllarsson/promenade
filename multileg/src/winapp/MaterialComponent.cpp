#include "MaterialComponent.h"

MaterialComponent::MaterialComponent()
{ 
	setColorRGB({ 1, 1, 1 }); 
	init();
}

MaterialComponent::MaterialComponent(const Color4f& p_color)
{ 
	setColorRGBA(p_color); 
	init();
}

MaterialComponent::MaterialComponent(const Color3f& p_color)
{ 
	setColorRGB(p_color); 
	init();
}


void MaterialComponent::setColorRGB(const Color3f& p_color)
{
	setColorRGBA(toColor4f(p_color));
}

void MaterialComponent::setColorRGBA(const Color4f& p_color)
{
	m_color = p_color;
	m_dirty = true;
}

const Color4f& MaterialComponent::getColorRGBA()
{ 
	return m_highLightFac?dawnBringerPalRGBA[COL_YELLOW]:m_color; 
}

const Color3f MaterialComponent::getColorRGB()
{ 
	return m_highLightFac?dawnBringerPalRGB[COL_YELLOW]:toColor3f(m_color); 
}

bool MaterialComponent::isMaterialRenderDirty()
{ 
	return m_dirty; 
}

void MaterialComponent::unsetMaterialRenderDirty()
{ 
	m_dirty = false; 
}


void MaterialComponent::highLight()
{
	m_highLightFac = true;
	m_dirty = true;
}

void MaterialComponent::init()
{
	m_dirty = false;
	m_highLightFac = false;
}

void MaterialComponent::unsetHighLight()
{
	m_dirty = true;
	m_highLightFac = false;
}
