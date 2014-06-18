#include "Toolbar.h"
#include <AntTweakBar.h>
#include <ToString.h>


Toolbar::Toolbar(void* p_device)
{
	TwInit(TW_DIRECT3D11, p_device);

	m_bars.push_back(Bar("PERFORMANCE"));
	m_bars.push_back(Bar("CHARACTER"));

	//TwDefine(" GLOBAL fontscaling=0.5 "); // font size must be set before init (exception)
	init();
}

Toolbar::~Toolbar()
{
	TwTerminate();
}

void Toolbar::init()
{
	TwDefine(" GLOBAL contained=true ");
	TwDefine(" GLOBAL fontsize=1 ");
	TwDefine(" GLOBAL fontresizable=false ");
	TwDefine(" GLOBAL refresh=0.25 ");
	defineBarParams(PERFORMANCE, dawnBringer32Pal[DB32COL_DARKPURPLE], " position= '0 0' size='200 150' ");
	defineBarParams(CHARACTER, dawnBringer32Pal[DB32COL_CONCRETE], " position= '0 160' size='200 400' ");
}

int Toolbar::process(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (TwEventWin(wnd, msg, wParam, lParam)) // send event message to AntTweakBar
		return 0; // event has been handled by AntTweakBar
}


void Toolbar::setWindowSize(int p_width, int p_height)
{
	TwWindowSize(p_width, p_height);
}

void Toolbar::draw()
{
	TwDraw();
}

void Toolbar::defineBarParams(BarType p_type, const char* p_params)
{
	std::string tmp = " ";
	tmp += m_bars[(int)p_type].m_name;
	tmp += p_params;
	TwDefine(tmp.c_str());
}

void Toolbar::defineBarParams(BarType p_type, const Color3& p_color, const char* p_params)
{
	std::string colStr = " color='";
	colStr += ToString((int)(256.0f*p_color.r))+" ";
	colStr += ToString((int)(256.0f*p_color.g)) + " ";
	colStr += ToString((int)(256.0f*p_color.b));
	colStr += "' ";
	defineBarParams(p_type, (colStr + std::string(p_params)).c_str());
}

void Toolbar::defineBarParams(BarType p_type, float p_min, float p_max, float p_stepSz, const char* p_params)
{
	std::string paramStr = " min='";
	paramStr += ToString(p_min) + " ";
	paramStr += "' max=";
	paramStr += ToString(p_max) + " ";
	paramStr += "' step=";
	paramStr += ToString(p_stepSz);
	paramStr += "' ";
	defineBarParams(p_type, (paramStr + std::string(p_params)).c_str());
}

void Toolbar::defineBarParams(BarType p_type, int p_min, int p_max, const char* p_params)
{
	std::string paramStr = " min='";
	paramStr += ToString(p_min) + " ";
	paramStr += "' max=";
	paramStr += ToString(p_max);
	paramStr += "' ";
	defineBarParams(p_type, (paramStr + std::string(p_params)).c_str());
}

void Toolbar::addReadOnlyVariable(BarType p_barType, const char* p_name, VarType p_type, const void *p_var, const char* p_misc)
{
	TwAddVarRO(getBar(p_barType), p_name, (TwType)p_type, p_var, p_misc);
}

void Toolbar::addWriteVariable(BarType p_barType, const char* p_name, VarType p_type, void *p_var, const char* p_misc)
{
	TwAddVarRW(getBar(p_barType), p_name, (TwType)p_type, p_var, p_misc);
}

void Toolbar::addSeparator(BarType p_barType, const char* p_name, VarType p_type, const char* p_misc)
{

}

void Toolbar::addButton(BarType p_barType, const char* p_name, TwButtonCallback p_callback, void *p_inputData, const char* p_misc)
{
	TwAddButton(getBar(p_barType), p_name, p_callback, p_inputData, p_misc);
}

void Toolbar::addLabel(BarType p_barType, const char* p_name, const char* p_label)
{
	TwAddButton(getBar(p_barType), p_name, NULL, NULL, (std::string(" label='")+std::string(p_label)+"'").c_str() );
}

TwBar* Toolbar::getBar(BarType p_type)
{
	return m_bars[(int)p_type].m_bar;
}
