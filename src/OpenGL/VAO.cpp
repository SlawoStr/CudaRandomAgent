#include "VAO.h"


////////////////////////////////////////////////////////////
VAO::VAO()
{
	glGenVertexArrays(1, &ID);
}

////////////////////////////////////////////////////////////
void VAO::LinkAttrib(VBO vbo, GLuint layout, GLuint numComponenets, GLenum type, GLsizei stride, void* offset)
{
	vbo.Bind();
	glVertexAttribPointer(layout, numComponenets, type, GL_FALSE, stride, offset);
	glEnableVertexAttribArray(layout);
	vbo.Unbind();
}

////////////////////////////////////////////////////////////
void VAO::Bind()
{
	glBindVertexArray(ID);
}

////////////////////////////////////////////////////////////
void VAO::Unbind()
{
	glBindVertexArray(0);
}

////////////////////////////////////////////////////////////
void VAO::Delete()
{
	glDeleteVertexArrays(1, &ID);
}

