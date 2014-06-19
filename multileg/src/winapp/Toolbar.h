#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <AntTweakBar.h>
#include <ColorPalettes.h>
#include <IContextProcessable.h>

// =======================================================================================
//                                      Toolbar
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Wrapper for AntTweakBar
///        
/// # Toolbar
/// 
/// 18-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class Toolbar : public IContextProcessable
{
public:
	class Bar
	{
	public:
		Bar(const std::string& p_name)
		{
			m_name = p_name;
			m_bar = TwNewBar(m_name.c_str());
		}
		~Bar()
		{ 
			//delete m_bar; 
		}
		std::string m_name;
		TwBar* m_bar;
	};
	
	enum BarType
	{
		PLAYER,
		PERFORMANCE,
		CHARACTER
	};

	enum VarType
	{
		FLOAT = TwType::TW_TYPE_FLOAT,
		DOUBLE = TwType::TW_TYPE_DOUBLE,
		BOOL = TwType::TW_TYPE_BOOLCPP,
		INT = TwType::TW_TYPE_INT32,
		UNSIGNED_INT = TwType::TW_TYPE_UINT32,
		SHORT = TwType::TW_TYPE_INT16,
		UNSIGNED_SHORT = TwType::TW_TYPE_UINT16,
		VEC3 = TwType::TW_TYPE_DIR3F,
		DIR = TwType::TW_TYPE_DIR3F,
		QUAT = TwType::TW_TYPE_QUAT4F,
		COL_RGB = TwType::TW_TYPE_COLOR3F,
		COL_RGBA = TwType::TW_TYPE_COLOR4F,
		STRING = TwType::TW_TYPE_STDSTRING
	};


	Toolbar(void* p_device);
	virtual ~Toolbar();

	void init();

	virtual bool processEvent(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	void setWindowSize(int p_width, int p_height);
	void draw();

	void addReadOnlyVariable(BarType p_barType, const char* p_name, VarType p_type,
		const void *p_var, const char* p_misc="");

	void addReadWriteVariable(BarType p_barType, const char* p_name, VarType p_type, void *p_var,
		const char* p_misc = "");

	void addSeparator(BarType p_barType, const char* p_name,
		const char* p_misc = "");

	void addButton(BarType p_barType, const char* p_name, TwButtonCallback p_callback, void *p_inputData,
		const char* p_misc = "");

	void addLabel(BarType p_barType, const char* p_name, const char* p_misc="");

	void defineBarParams(BarType p_type, const char* p_params);
	void defineBarParams(BarType p_type, float p_min, float p_max, float p_stepSz, const char* p_params);
	void defineBarParams(BarType p_type, int p_min, int p_max, const char* p_params);
	void defineBarParams(BarType p_type, const Color3f& p_color, const char* p_params);

	TwBar* getBar(BarType p_type);

protected:
private:
	std::vector<Bar> m_bars;
};

void TW_CALL boolButton(void* p_bool);