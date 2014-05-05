#pragma once

// =======================================================================================
//                                      Raw Texture
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Texture struct, raw data
///        
/// # Texture
/// 
/// 5-4-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

struct RawTexture
{
	RawTexture(float* p_data, 
			   unsigned int p_width, unsigned int p_height, 
			   unsigned int p_channels)
	{
		m_data=p_data;
		m_width=p_width;
		m_height=p_height;
		m_channels=p_channels;
	}
	float* m_data;
	unsigned int m_width, m_height, m_channels; // size is width*height*channel, pitch is channels
};