#pragma once
#include <glad/glad.h>
#include "VBO.h"

/// <summary>
/// VAO Class
/// </summary>
class VAO
{
public:
	VAO();

	/// <summary>
	/// Link atributes 
	/// </summary>
	/// <param name="vbo">VBO</param>
	/// <param name="layout">Starting point of data</param>
	/// <param name="numComponenets">Number of elements</param>
	/// <param name="type">Data type</param>
	/// <param name="stride">Memory stride of data</param>
	/// <param name="offset">Data offset</param>
	void LinkAttrib(VBO vbo, GLuint layout, GLuint numComponenets, GLenum type, GLsizei stride, void* offset);

	/// <summary>
	/// BIND VAO
	/// </summary>
	void Bind();

	/// <summary>
	/// Unbind VAO
	/// </summary>
	void Unbind();

	/// <summary>
	/// Delete VAO
	/// </summary>
	void Delete();

public:
	GLuint ID;		//!< VAO ID
};