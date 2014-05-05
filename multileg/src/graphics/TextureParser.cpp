#include "TextureParser.h"
#include <FreeImage.h>
#include <Util.h>
#include <vector>
#include <DebugPrint.h>

void TextureParser::init()
{
	FreeImage_Initialise();
}

ID3D11ShaderResourceView* TextureParser::loadTexture(ID3D11Device* p_device, 
													 ID3D11DeviceContext* p_context,
													 const char* p_filePath)
{
	FREE_IMAGE_FORMAT imageFormat;
	FIBITMAP* image;
	bool succeededLoadingFile = true;
	ID3D11ShaderResourceView* newShaderResurceView;

	imageFormat = FreeImage_GetFIFFromFilename(p_filePath);
	if( imageFormat != FIF_UNKNOWN )
		image = FreeImage_Load(imageFormat, p_filePath);
	else
	{
		/************************************************************************/
		/* Made reverting back to a fallback texture will be enough? Instead	*/
		/* of throwing a exception.												*/
		/************************************************************************/
		succeededLoadingFile = false;
		DEBUGWARNING(((string("Unknown texture file format, cannot parse the file, reverting to fallback texture. ") + 
					  toString(p_filePath)).c_str()));
	}
	/************************************************************************/
	/* If the width of the image is equal to zero, then the file wasn't		*/
	/* found																*/
	/************************************************************************/
	if (succeededLoadingFile && FreeImage_GetWidth(image)==0)
	{
		DEBUGWARNING(((string("Texture file not found, reverting to fallback texture. ") + 
			toString(p_filePath)).c_str()));
		succeededLoadingFile = false;
	}
	if(succeededLoadingFile)
	{
		FreeImage_FlipVertical(image);

		newShaderResurceView = createTexture(
			p_device, p_context, FreeImage_GetBits(image), FreeImage_GetWidth(image),
			FreeImage_GetHeight(image), FreeImage_GetPitch(image), FreeImage_GetBPP(image),
			TextureParser::TEXTURE_TYPE::BGRA);

		/************************************************************************/
		/* Clean up the mess afterwards											*/
		/************************************************************************/
		FreeImage_Unload(image);
	}
	else
	{
		BYTE* data = generateFallbackTexture();
		newShaderResurceView = createTexture(p_device, p_context, data,10,10,128,32,
			TextureParser::TEXTURE_TYPE::RGBA);

		delete data;
	}
	return newShaderResurceView;
}

RawTexture* TextureParser::loadTexture( const char* p_filePath )
{
	FREE_IMAGE_FORMAT imageFormat;
	FIBITMAP* image;
	bool succeededLoadingFile = true;
	RawTexture* texture;

	imageFormat = FreeImage_GetFIFFromFilename(p_filePath);
	if( imageFormat != FIF_UNKNOWN )
		image = FreeImage_Load(imageFormat, p_filePath);
	else
	{
		// Revert to fallback if texture unknown
		succeededLoadingFile = false;
		DEBUGWARNING(((string("Unknown texture file format, cannot parse the file, reverting to fallback texture. ") + 
			toString(p_filePath)).c_str()));
	}
	// If width==0, we cannot use the texture, also revert to fallback
	if (succeededLoadingFile && FreeImage_GetWidth(image)==0)
	{
		DEBUGWARNING(((string("Texture file not found, reverting to fallback texture. ") + 
			toString(p_filePath)).c_str()));
		succeededLoadingFile = false;
	}
	if(succeededLoadingFile)
	{
		//FreeImage_FlipVertical(image);

		 texture = createTexture(image, FreeImage_GetWidth(image),
			FreeImage_GetHeight(image));

		/************************************************************************/
		/* Clean up the mess afterwards											*/
		/************************************************************************/
		FreeImage_Unload(image);
	}
	else
	{
		float* input;
		int ww=656, hh=480;
		input = new float[ww*hh*4];		
		for(int y = 0; y < hh; y++)
		for(int x = 0; x < ww; x++)
		{
			unsigned int i=y*ww*4+x*4;
			// r
			input[i] = (float)i/(float)(ww*hh*4);
			// g
			input[i+1] = (1.0f-((float)i/(float)(ww*hh*4)));
			// b
			input[i+2] = 0.5f;
			// a
			input[i+3] = 1.0f;
		}
		texture = new RawTexture(input,ww,hh,4);
	}
	return texture;
}

ID3D11ShaderResourceView* TextureParser::createTexture( ID3D11Device* p_device, 
													   ID3D11DeviceContext* p_context, 
													   const byte* p_source, int p_width, 
													   int p_height, int p_pitch, 
													   int p_bitLevel, TEXTURE_TYPE p_type )
{
	int width = p_width;
	int height = p_height;
	int numOfMipLevels = 1;
	while ((width > 1) || (height > 1))
	{
		width = max(width / 2, 1);
		height = max(height / 2, 1);
		++numOfMipLevels;
	}

	byte* newData = NULL;

	if(p_bitLevel == 24){

		newData = new byte[p_width*p_height*4];
		unsigned int ind = 0;
		unsigned int counter = 0;
		
		for (int i = 0; i < p_width * p_height*4;i++){
			if(counter < 3){
				newData[i] = p_source[ind++];
				counter++;
			}
			else{
				newData[i] = 255;
				counter = 0;
			}
		}
	}

	D3D11_SUBRESOURCE_DATA* data = new D3D11_SUBRESOURCE_DATA[numOfMipLevels];
	ZeroMemory(&data[0], sizeof(D3D11_SUBRESOURCE_DATA));
	if(newData){
		data[0].pSysMem = (void*)newData;
	}
	else{
		data[0].pSysMem = (void*)p_source;
	}
	data[0].SysMemPitch = p_width*4;
	data[0].SysMemSlicePitch = 0;

	width = p_width;
	height = p_height;
	std::vector<unsigned char*> levelData;
	for(int i = 1; i< numOfMipLevels; i++)
	{
		ZeroMemory(&data[i], sizeof(D3D11_SUBRESOURCE_DATA));
		width = static_cast<int>(max(1.0f, width*0.5f));
		height = static_cast<int>(max(1.0f, height*0.5f));
		levelData.push_back(new unsigned char[4*width*height]);
		data[i].pSysMem = levelData.back();
		data[i].SysMemPitch = 4 * width;
		data[i].SysMemSlicePitch = 0;
	} 

	D3D11_TEXTURE2D_DESC texDesc;
	ZeroMemory(&texDesc,sizeof(D3D11_TEXTURE2D_DESC));
	texDesc.Width				= p_width;
	texDesc.Height				= p_height;
	texDesc.MipLevels			= numOfMipLevels;
	texDesc.ArraySize			= 1;
	texDesc.SampleDesc.Count	= 1;
	texDesc.SampleDesc.Quality	= 0;
	texDesc.Usage				= D3D11_USAGE_DEFAULT;
	texDesc.BindFlags			= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	texDesc.CPUAccessFlags		= 0;
	texDesc.MiscFlags			= D3D11_RESOURCE_MISC_GENERATE_MIPS;

	switch( p_type )
	{
	case RGBA:
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		break;		   

	case BGRA:		   
		texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		break;		   

	default:		   
		texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		break;
	}

	ID3D11Texture2D* texture = NULL;
	HRESULT hr = p_device->CreateTexture2D( &texDesc, data, &texture);

	for (unsigned int i = 0; i < levelData.size(); i++)
		delete[] levelData[i];

	if (FAILED(hr))
		throw GraphicsException(hr, __FILE__,__FUNCTION__,__LINE__);

	D3D11_SHADER_RESOURCE_VIEW_DESC shaderDesc;
	ZeroMemory(&shaderDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	shaderDesc.Format = texDesc.Format;
	shaderDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shaderDesc.Texture2D.MipLevels = numOfMipLevels;
	shaderDesc.Texture2D.MostDetailedMip = 0;

	ID3D11ShaderResourceView* newShaderResurceView;
	hr = p_device->CreateShaderResourceView( texture, &shaderDesc, 
		&newShaderResurceView);

	if (FAILED(hr))
		throw GraphicsException(hr, __FILE__,__FUNCTION__,__LINE__);

	p_context->GenerateMips(newShaderResurceView);

	texture->Release();
	delete [] newData;
	delete [] data;

	return newShaderResurceView;
}

RawTexture* TextureParser::createTexture( FIBITMAP* p_bitmap, int p_width, int p_height)
{
	int width = p_width;
	int height = p_height;

	float* newData = NULL;

	unsigned int channels=4;
	unsigned int size=width*height*channels;
	newData = new float[size];
	vector<float> g;

	for (unsigned int y=0;y<height;y++)
	for (unsigned int x=0;x<width;x++)
	{
		unsigned int idx=y*width*channels+x*channels;
		RGBQUAD color;
		bool res=FreeImage_GetPixelColor(p_bitmap, x, y, &color)==0?false:true;
		if (!res)
			throw GraphicsException("Bitmap was parsed incorrectly! ",__FILE__,__FUNCTION__,__LINE__);

		newData[idx]=((float)color.rgbRed/256.0f);		g.push_back(newData[idx]);
		newData[idx+1]=((float)color.rgbGreen/256.0f);	g.push_back(newData[idx+1]);
		newData[idx+2]=((float)color.rgbBlue/256.0f);	g.push_back(newData[idx+2]);
		newData[idx+3]=1.0f;							g.push_back(newData[idx+3]);
	}
	RawTexture* tex=new RawTexture(newData,width,height,channels);
	return tex;
}


BYTE* TextureParser::generateFallbackTexture()
{
	int dimension = 10;
	int size = dimension*dimension;
	int bitDepth = 4;
	BYTE* textureData = new BYTE[size*bitDepth];
	bool pink = true;

	// Generate xy gradient
	for (int y = 0; y < dimension; y++)
	{
		for (int x = 0; x < dimension; x++)
		{
			textureData[dimension*y*bitDepth + x*bitDepth]		= (int)((float)x/(float)dimension)*255;		//RED
			textureData[dimension*y*bitDepth + x*bitDepth+1]	= (int)((float)y/(float)dimension)*255;		//BLUE
			textureData[dimension*y*bitDepth + x*bitDepth+2]	= 0;	//GREEN
			textureData[dimension*y*bitDepth + x*bitDepth+3]	= 255;	//ALPHA
		}
	}
	
	return textureData;
}
